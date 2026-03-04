
from __future__ import annotations
import os, io, json, math, time, hashlib, sqlite3, contextlib, pathlib
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Body
from pydantic import BaseModel, Field

# Optional but recommended for parquet output
try:
    import pyarrow as pa  # noqa: F401
    import pyarrow.parquet as pq
    _HAS_PARQUET = True
except Exception:
    _HAS_PARQUET = False

# ------------------------------
# Utilities & Cache
# ------------------------------
DATA_DIR = pathlib.Path(os.environ.get("DATA_DIR", "/data")).resolve()
CACHE_DB = DATA_DIR / "cache.sqlite3"
DATASETS_DIR = DATA_DIR / "datasets"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

class DiskCache:
    def __init__(self, db_path: pathlib.Path):
        self.db_path = str(db_path)
        with sqlite3.connect(self.db_path) as con:
            con.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v BLOB, ts REAL)")
            con.commit()

    def _key(self, name: str, args: Dict[str, Any]) -> str:
        payload = json.dumps({"name": name, "args": args}, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, name: str, args: Dict[str, Any], ttl_seconds: Optional[int] = None) -> Optional[Any]:
        k = self._key(name, args)
        with sqlite3.connect(self.db_path) as con:
            cur = con.execute("SELECT v, ts FROM cache WHERE k=?", (k,))
            row = cur.fetchone()
            if not row:
                return None
            v, ts = row
            if ttl_seconds is not None and (time.time() - ts) > ttl_seconds:
                return None
            try:
                return json.loads(v)
            except Exception:
                return None

    def set(self, name: str, args: Dict[str, Any], value: Any) -> str:
        k = self._key(name, args)
        with sqlite3.connect(self.db_path) as con:
            con.execute("REPLACE INTO cache(k, v, ts) VALUES (?,?,?)", (k, json.dumps(value, default=str), time.time()))
            con.commit()
        return k

CACHE = DiskCache(CACHE_DB)

# ------------------------------
# Provider Interface & Default (yfinance)
# ------------------------------
class DataProvider:
    def get_ohlcv(self, symbol: str, timeframe: str, start: Optional[str], end: Optional[str], adj: bool) -> pd.DataFrame:
        raise NotImplementedError

    def get_options_chain(self, underlier: str, expiry: Optional[str], strike_range: Optional[List[float]], right: Optional[str]) -> pd.DataFrame:
        raise NotImplementedError

    def get_corporate_events(self, symbol: str, start: Optional[str], end: Optional[str]) -> Dict[str, Any]:
        raise NotImplementedError

    def get_iv_surface(self, underlier: str, at: Optional[str]) -> pd.DataFrame:
        raise NotImplementedError

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

class YFinanceProvider(DataProvider):
    def __init__(self):
        import yfinance as yf
        self.yf = yf

    @staticmethod
    def _parse_interval(tf: str) -> str:
        # Map common tf to yfinance intervals
        m = {
            "1m":"1m","2m":"2m","5m":"5m","15m":"15m","30m":"30m","60m":"60m","90m":"90m",
            "1h":"60m","1d":"1d","1wk":"1wk","1mo":"1mo"
        }
        if tf not in m:
            raise HTTPException(400, f"Unsupported timeframe: {tf}")
        return m[tf]

    def get_ohlcv(self, symbol: str, timeframe: str, start: Optional[str], end: Optional[str], adj: bool) -> pd.DataFrame:
        iv = self._parse_interval(timeframe)
        ticker = self.yf.Ticker(symbol)
        df = ticker.history(interval=iv, start=start, end=end, auto_adjust=adj)
        if df is None or df.empty:
            raise HTTPException(404, f"No OHLCV for {symbol}")
        df = df.rename(columns={"Open":"o","High":"h","Low":"l","Close":"c","Volume":"v"})
        df.index = pd.to_datetime(df.index, utc=True)
        return df[["o","h","l","c","v"]]

    def get_options_chain(self, underlier: str, expiry: Optional[str], strike_range: Optional[List[float]], right: Optional[str]) -> pd.DataFrame:
        t = self.yf.Ticker(underlier)
        expiries = t.options or []
        if not expiries:
            raise HTTPException(404, f"No options listed for {underlier}")
        if expiry and expiry not in expiries:
            raise HTTPException(400, f"Expiry {expiry} not in available expiries: {expiries[:8]}…")
        target_exps = [expiry] if expiry else expiries[:1]
        frames = []
        for e in target_exps:
            chain = t.option_chain(e)
            for side, df in [("call", chain.calls), ("put", chain.puts)]:
                df = df.copy()
                df["right"] = side
                df["expiry"] = e
                frames.append(df)
        out = pd.concat(frames, ignore_index=True)
        if strike_range:
            lo, hi = (min(strike_range), max(strike_range))
            out = out[(out["strike"] >= lo) & (out["strike"] <= hi)]
        if right in ("call","put"):
            out = out[out["right"] == right]
        # Normalize columns
        rename = {
            "contractSymbol":"contract",
            "lastPrice":"last",
            "impliedVolatility":"iv",
            "inTheMoney":"itm"
        }
        out = out.rename(columns=rename)
        out["iv"] = out["iv"].astype(float).replace(0, np.nan)
        return out

    def get_corporate_events(self, symbol: str, start: Optional[str], end: Optional[str]) -> Dict[str, Any]:
        t = self.yf.Ticker(symbol)
        # Earnings dates
        try:
            cal = t.get_earnings_dates(limit=12)
        except Exception:
            cal = None
        divs = t.dividends
        splits = t.splits
        def _frame(df: Optional[pd.Series | pd.DataFrame]):
            if df is None or len(df) == 0:
                return []
            if isinstance(df, pd.Series):
                df = df.to_frame()
            df.index = pd.to_datetime(df.index, utc=True)
            if start:
                df = df[df.index >= pd.Timestamp(start, tz="UTC")]
            if end:
                df = df[df.index <= pd.Timestamp(end, tz="UTC")]
            return [
                {"t": i.isoformat(), "fields": {k: (float(v) if pd.notna(v) else None) for k, v in row.items()}}
                for i, row in df.iterrows()
            ]
        events = {
            "earnings": _frame(cal),
            "dividends": _frame(divs),
            "splits": _frame(splits),
        }
        return events

    def get_iv_surface(self, underlier: str, at: Optional[str]) -> pd.DataFrame:
        # Build a simple surface from near expiry chains
        oc = self.get_options_chain(underlier, expiry=None, strike_range=None, right=None)
        oc = oc[["expiry","right","strike","iv"]].dropna()
        # time to expiry (year fraction) using current date
        now = pd.Timestamp(at, tz="UTC") if at else pd.Timestamp.now(tz="UTC")
        oc["T"] = (pd.to_datetime(oc["expiry"]).dt.tz_localize("UTC") - now).dt.total_seconds() / (365.0*24*3600)
        oc = oc[oc["T"] > 0]
        return oc

PROVIDER: DataProvider = YFinanceProvider()

# ------------------------------
# Greeks (Black–Scholes) helpers
# ------------------------------
from math import log, sqrt, exp
from scipy.stats import norm  # heavy but convenient; acceptable for a service

def _bs_d1(S, K, r, sigma, T):
    return (log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))

def _bs_d2(d1, sigma, T):
    return d1 - sigma*sqrt(T)

def greeks(S, K, r, sigma, T, right: str):
    # returns delta, gamma, theta, vega, rho
    if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
        return {"delta": None, "gamma": None, "theta": None, "vega": None, "rho": None}
    d1 = _bs_d1(S, K, r, sigma, T)
    d2 = _bs_d2(d1, sigma, T)
    if right == "call":
        delta = norm.cdf(d1)
        theta = -(S*norm.pdf(d1)*sigma/(2*sqrt(T))) - r*K*exp(-r*T)*norm.cdf(d2)
        rho = K*T*exp(-r*T)*norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1
        theta = -(S*norm.pdf(d1)*sigma/(2*sqrt(T))) + r*K*exp(-r*T)*norm.cdf(-d2)
        rho = -K*T*exp(-r*T)*norm.cdf(-d2)
    gamma = norm.pdf(d1)/(S*sigma*sqrt(T))
    vega = S*norm.pdf(d1)*sqrt(T)
    return {"delta": float(delta), "gamma": float(gamma), "theta": float(theta), "vega": float(vega), "rho": float(rho)}

# ------------------------------
# Pydantic Schemas
# ------------------------------
class OHLCVOut(BaseModel):
    t: str; o: float; h: float; l: float; c: float; v: float

class OptionRecord(BaseModel):
    contract: str
    right: str
    expiry: str
    strike: float
    last: Optional[float] = None
    iv: Optional[float] = None
    itm: Optional[bool] = None

class GreeksIn(BaseModel):
    contracts: List[str] = Field(..., description="Exact option symbols; if omitted, provide underlier+expiry+strike+right in 'details'.")
    details: Optional[List[Dict[str, Any]]] = Field(None, description="Per-contract details if symbol not resolvable via provider")
    at: Optional[str] = None
    r: float = 0.00

class IVSurfaceOut(BaseModel):
    expiry: str
    right: str
    strike: float
    iv: float
    T: float

class MakeDatasetIn(BaseModel):
    symbols: List[str]
    features: List[str]
    horizon: str = Field(..., description="Labeling horizon, e.g., '1d' for next-day move")
    window: str = Field(..., description="Lookback window to gather features, e.g., '120d'")
    align: str = Field("market_close", description="market_open|market_close alignment for labels")

class DatasetOut(BaseModel):
    uri: str
    rows: int
    cols: int
    key: str

# ------------------------------
# FastAPI app
# ------------------------------
app = FastAPI(title="market-data MCP", version="0.1.0")

@app.get("/health")
def health():
    return {"ok": True, "now": PROVIDER.now().isoformat()}

@app.get("/get_ohlcv", response_model=List[OHLCVOut])
def get_ohlcv(symbol: str = Query(...), timeframe: str = Query("1d"), start: Optional[str] = None, end: Optional[str] = None, adj: bool = True):
    args = {"symbol": symbol, "timeframe": timeframe, "start": start, "end": end, "adj": adj}
    cached = CACHE.get("get_ohlcv", args, ttl_seconds=3600)
    if cached is not None:
        return cached
    df = PROVIDER.get_ohlcv(symbol, timeframe, start, end, adj)
    out = [{"t": i.isoformat(), **row._asdict()} if hasattr(row, "_asdict") else {"t": i.isoformat(), **{k: float(row[k]) for k in ["o","h","l","c","v"]}} for i, row in df.iterrows()]
    CACHE.set("get_ohlcv", args, out)
    return out

@app.get("/get_options_chain", response_model=List[OptionRecord])
def get_options_chain(underlier: str, expiry: Optional[str] = None, strike_min: Optional[float] = None, strike_max: Optional[float] = None, right: Optional[str] = Query(None, pattern="^(call|put)$")):
    strike_range = None
    if strike_min is not None or strike_max is not None:
        lo = float(strike_min if strike_min is not None else -np.inf)
        hi = float(strike_max if strike_max is not None else np.inf)
        strike_range = [lo, hi]
    args = {"underlier": underlier, "expiry": expiry, "strike_range": strike_range, "right": right}
    cached = CACHE.get("get_options_chain", args, ttl_seconds=900)
    if cached is not None:
        return cached
    df = PROVIDER.get_options_chain(underlier, expiry, strike_range, right)
    keep = ["contract","right","expiry","strike","last","iv","itm"]
    df = df[keep]
    out = json.loads(df.to_json(orient="records"))
    CACHE.set("get_options_chain", args, out)
    return out

@app.post("/get_greeks")
def get_greeks(payload: GreeksIn):
    # We support two modes: by known option symbols (provider-specific) or by explicit details.
    at = pd.Timestamp(payload.at, tz="UTC") if payload.at else pd.Timestamp.now(tz="UTC")
    r = float(payload.r)
    results = []
    # Minimal spot lookup via yfinance last close
    spots: Dict[str, float] = {}

    # If details provided, we bypass symbol parsing
    if payload.details:
        for d in payload.details:
            S = float(d["spot"]) if "spot" in d else None
            if S is None:
                und = d.get("underlier")
                if und and und not in spots:
                    ohlc = PROVIDER.get_ohlcv(und, "1d", None, None, True)
                    spots[und] = float(ohlc["c"].iloc[-1])
                S = spots.get(und)
            K = float(d["strike"]) ; right = d["right"] ; expiry = pd.Timestamp(d["expiry"]).tz_localize("UTC")
            T = max((expiry - at).total_seconds(), 0.0)/(365*24*3600)
            sigma = float(d.get("iv", np.nan))
            if not np.isfinite(sigma):
                sigma = 0.0
            g = greeks(S, K, r, sigma, T, right)
            results.append({"contract": d.get("contract","unknown"), **g})
        return results

    # Symbol-only path (best-effort using provider option chain to fetch iv, strike, expiry & underlier spot)
    for oc_symbol in payload.contracts:
        # Try to infer underlier from typical OCC format: e.g., AAPL250117C00180000
        try:
            # crude parse: letters prefix underlier until first digit
            i = next(idx for idx, ch in enumerate(oc_symbol) if ch.isdigit())
            und = oc_symbol[:i]
        except Exception:
            und = None
        if und and und not in spots:
            ohlc = PROVIDER.get_ohlcv(und, "1d", None, None, True)
            spots[und] = float(ohlc["c"].iloc[-1])
        S = spots.get(und)
        # Without a full symbol resolver, return nulls but keep structure
        results.append({"contract": oc_symbol, "delta": None, "gamma": None, "theta": None, "vega": None, "rho": None, "spot": S})
    return results

@app.get("/get_iv_surface", response_model=List[IVSurfaceOut])
def get_iv_surface(underlier: str, at: Optional[str] = None):
    args = {"underlier": underlier, "at": at}
    cached = CACHE.get("get_iv_surface", args, ttl_seconds=900)
    if cached is not None:
        return cached
    df = PROVIDER.get_iv_surface(underlier, at)
    keep = ["expiry","right","strike","iv","T"]
    df = df[keep].dropna()
    out = json.loads(df.to_json(orient="records"))
    CACHE.set("get_iv_surface", args, out)
    return out

@app.get("/get_corporate_events")
def get_corporate_events(symbol: str, start: Optional[str] = None, end: Optional[str] = None):
    args = {"symbol": symbol, "start": start, "end": end}
    cached = CACHE.get("get_corporate_events", args, ttl_seconds=3600)
    if cached is not None:
        return cached
    out = PROVIDER.get_corporate_events(symbol, start, end)
    CACHE.set("get_corporate_events", args, out)
    return out

# ------------------------------
# Dataset builder (feature engineering + alignment)
# ------------------------------
# Supported feature tokens (extend as needed):
#   - ohlcv(INTERVAL, LOOKBACK)
#   - ret_1d, ret_5d
#   - rv_park(Nd): Parkinson RV over N days
#   - iv30: 30d average of at-the-money IV (approx from mid strikes)

class _FeatSpec(BaseModel):
    kind: str
    interval: Optional[str] = None
    lookback: Optional[str] = None
    param: Optional[str] = None


def _parse_feature_token(tok: str) -> _FeatSpec:
    tok = tok.strip()
    if tok.startswith("ohlcv(") and tok.endswith(")"):
        inside = tok[6:-1]
        parts = [p.strip() for p in inside.split(",")]
        if len(parts) != 2:
            raise HTTPException(400, f"ohlcv expects 2 params, got {inside}")
        return _FeatSpec(kind="ohlcv", interval=parts[0], lookback=parts[1])
    if tok.startswith("rv_park(") and tok.endswith(")"):
        inside = tok[8:-1]
        return _FeatSpec(kind="rv_park", param=inside)
    if tok in ("ret_1d","ret_5d","iv30"):
        return _FeatSpec(kind=tok)
    raise HTTPException(400, f"Unknown feature token: {tok}")


def _to_days(s: str) -> int:
    s = s.strip().lower()
    if s.endswith("d"):
        return int(float(s[:-1]))
    if s.endswith("wk"):
        return int(7*float(s[:-2]))
    if s.endswith("mo"):
        return int(30*float(s[:-2]))
    raise HTTPException(400, f"Unsupported duration: {s}")


@app.post("/make_dataset", response_model=DatasetOut)
def make_dataset(spec: MakeDatasetIn = Body(...)):
    # Deterministic cache key
    args = spec.dict()
    cached = CACHE.get("make_dataset", args, ttl_seconds=None)
    if cached is not None and (DATASETS_DIR / pathlib.Path(cached["uri"]).name).exists():
        return cached

    feats = [_parse_feature_token(f) for f in spec.features]
    max_lookback_days = 0
    for f in feats:
        if f.lookback:
            max_lookback_days = max(max_lookback_days, _to_days(f.lookback))
        if f.kind in ("ret_5d", "iv30"):
            max_lookback_days = max(max_lookback_days, 30)
        if f.kind == "ret_1d":
            max_lookback_days = max(max_lookback_days, 2)

    end_dt = PROVIDER.now()
    start_dt = end_dt - timedelta(days=max_lookback_days + 5)

    frames = []
    for sym in spec.symbols:
        base = PROVIDER.get_ohlcv(sym, "1d", start_dt.date().isoformat(), end_dt.date().isoformat(), True)
        base = base.copy()
        base["symbol"] = sym
        # Basic returns
        base["ret_1d"] = base["c"].pct_change(1)
        base["ret_5d"] = base["c"].pct_change(5)
        # Parkinson RV over N days (if requested)
        def _rv_park(N: int):
            hl = np.log(base["h"]) - np.log(base["l"])
            rv = (hl**2).rolling(N).mean() / (4*np.log(2))
            return rv
        # iv30: average of mid-strike IV from nearest expiry (approx)
        def _iv30():
            try:
                oc = PROVIDER.get_options_chain(sym, expiry=None, strike_range=None, right=None)
                if oc is None or len(oc)==0:
                    return pd.Series(index=base.index, dtype=float)
                # Take nearest expiry and IV around ATM (within 5% of spot)
                spot = float(base["c"].iloc[-1])
                oc["moneyness"] = np.abs(oc["strike"] - spot)/spot
                near = oc.sort_values("expiry").groupby("expiry").head(200)
                near = near[near["moneyness"] <= 0.05]
                iv = float(near["iv"].dropna().mean()) if len(near)>0 else np.nan
                return pd.Series(iv, index=base.index)
            except Exception:
                return pd.Series(index=base.index, dtype=float)

        # Attach requested features
        feat_cols = {}
        for f in feats:
            if f.kind == "ret_1d":
                feat_cols["ret_1d"] = base["ret_1d"]
            elif f.kind == "ret_5d":
                feat_cols["ret_5d"] = base["ret_5d"]
            elif f.kind == "rv_park":
                N = _to_days(f.param or "5d")
                feat_cols[f"rv_park_{N}d"] = _rv_park(N)
            elif f.kind == "iv30":
                feat_cols["iv30"] = _iv30()
            elif f.kind == "ohlcv":
                # Already fetched at 1d resolution; if user asks another interval, fetch/merge
                if f.interval != "1d":
                    try:
                        intr = PROVIDER.get_ohlcv(sym, f.interval, start_dt.date().isoformat(), end_dt.date().isoformat(), True)
                        # daily bar features from intraday (e.g., mean v, hl range)
                        agg = intr.resample("1D").agg({"o":"first","h":"max","l":"min","c":"last","v":"sum"})
                        agg.columns = [f"{c}_{f.interval}" for c in agg.columns]
                        base = base.join(agg, how="left")
                    except Exception:
                        pass
                # else: base already has o,h,l,c,v at 1d
            else:
                pass
        # Final frame
        fdf = pd.DataFrame({"t": base.index, **{k: v.values for k, v in feat_cols.items()}, "symbol": sym, "c": base["c"].values})
        frames.append(fdf)

    ds = pd.concat(frames, ignore_index=True)

    # Labeling (simple next-day return sign) aligned at market_close
    if spec.align not in ("market_open","market_close"):
        raise HTTPException(400, "align must be market_open|market_close")
    ds = ds.sort_values(["symbol","t"]).reset_index(drop=True)
    ds["y_ret_1d"] = ds.groupby("symbol")["c"].pct_change(-1)
    ds["y_label"] = np.where(ds["y_ret_1d"] > 0, 1, 0)

    # Persist to parquet/csv and return URI
    ts_key = hashlib.sha256(json.dumps(args, sort_keys=True, default=str).encode()).hexdigest()[:12]
    fname = f"dataset_{ts_key}.parquet" if _HAS_PARQUET else f"dataset_{ts_key}.csv"
    fpath = DATASETS_DIR / fname
    if _HAS_PARQUET:
        ds.to_parquet(fpath, index=False)
    else:
        ds.to_csv(fpath, index=False)

    uri = f"file://{fpath}"
    out = {"uri": uri, "rows": int(ds.shape[0]), "cols": int(ds.shape[1]), "key": ts_key}
    CACHE.set("make_dataset", args, out)
    return out
