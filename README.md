# mcp-market-data
Market & Options Data MCP ("market-data") — FastAPI server

A deployable, minimal, vendor-pluggable service that exposes endpoints used by an LLM
or any client to fetch OHLCV, option chains, Greeks, basic corporate events, and to
assemble a training dataset with alignment/caching. Uses yfinance as a default
provider so it runs out-of-the-box; swap in Polygon/IEX/etc by implementing the
DataProvider interface below.

It can also proxy the lama option-dataset chart service when `LAMA_UI_SERVICE_URL`
is configured.

Run locally:
```
  pip install -r requirements.txt
  uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Docker:
```
  docker build -t mcp-market-data:edge .
  docker run -p 8000:8000 --name mcp-market-data mcp-market-data:edge
```

Example curl:
```
  curl "https://market-data.verumnomen.com/get_ohlcv?symbol=AAPL&timeframe=1d&start=2024-01-01&end=2024-06-30"
  curl "https://market-data.verumnomen.com/get_lama_option_chart?strategy_type=RET3&option_ticker=AAPL250117C00200000"
  curl -X POST https://market-data.verumnomen.com/make_dataset -H 'Content-Type: application/json' -d '{
    "symbols":["AAPL","NVDA"],
    "features":["ohlcv(1d,120d)","rv_park(5d)","ret_1d","iv30"],
    "horizon":"1d","window":"180d","align":"market_close"
  }'
```

To enable the lama proxy endpoint:

```bash
export LAMA_UI_SERVICE_URL="https://your-lama-ui-service.example.com"
```

### Notes
- Image will publish to `ghcr.io/pishnuke/mcp-market-data:edge` on `master`, plus a `sha-<short>` tag, and to `ghcr.io/pishnuke/mcp-market-data:<tag>` when you push a Git tag like `v0.1.0`.
- Ensure your repo is **public** or that consumers have permission to pull from GHCR. For private repos, consumers need a token.
- The GHCR repository name is lowercase; if your GitHub org/repo has uppercase, GHCR normalizes it.

Point your MCP client at `mcp.json` (or the running URL) and call tools like `/get_ohlcv`.
