from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.main import app


def test_get_lama_option_chart_proxies_remote_service(monkeypatch) -> None:
    expected_payload = {
        "strategy_type": "RET3",
        "experiment_name": "Dataset: Stock options [RET3]",
        "dataset_run_name": "dataset-v42",
        "option_ticker": "AAPL250117C00200000",
        "abstract_option_ticker": "AAPL250117*00200000",
        "option_side": "C",
        "underlying_ticker": "AAPL",
        "points": [{"date": "2026-01-11", "offset": 1, "option_price": 5.5, "underlying_price": 195.0, "marked_buy": True}],
    }

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(expected_payload).encode()

    captured = {}

    def _urlopen(request, timeout=0):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setenv("LAMA_UI_SERVICE_URL", "https://lama-ui.example.com")
    monkeypatch.setattr("app.main.urlopen", _urlopen)

    client = TestClient(app)
    response = client.get(
        "/get_lama_option_chart",
        params={"strategy_type": "RET3", "option_ticker": "AAPL250117C00200000"},
    )

    assert response.status_code == 200
    assert response.json() == expected_payload
    assert captured["url"] == (
        "https://lama-ui.example.com/api/option-chart"
        "?strategy_type=RET3&option_ticker=AAPL250117C00200000"
    )
    assert captured["timeout"] == 20
