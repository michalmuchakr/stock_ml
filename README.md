# Stock Prediction ML Engineer — Take‑Home Test

## Objective
Build a **short‑horizon stock movement predictor** that:
1. Asks for a **US ticker** (e.g., `AAPL`) and **timeframe** (`1m`, `15m`, `1h`, `4h`).
2. Fetches historical OHLCV from the **Twelve Data** Market API.
3. Engineers at least **two technical indicators** using **TA‑Lib**.
4. Trains **two models**: `RandomForestClassifier` and `XGBClassifier` (XGBoost).
5. **Backtests** a simple next‑bar strategy and reports metrics + charts.

> Treat this as a 2–4 hour exercise. Focus on correctness, clarity, and clean structure.
> You do **not** need to over‑optimize performance or tune hyperparameters heavily.

---

## Deliverables
- Working code in `src/` with clear structure.
- **Backtest metrics** (printed to console) and **charts** saved under `artifacts/`:
  - Cumulative return curve
  - Rolling Sharpe (optional)
- A **short note** in `NOTES.md` describing design choices/tradeoffs and how to run.
- Use environment variable **`TWELVEDATA_API_KEY`** (do not hardcode keys).

---

## Data & Labels
- Use Twelve Data **Time Series** endpoint.
- Use **time‑ordered split** (no shuffling). Suggested: last 20% as test set.
- Define binary label: **next bar up = 1** if next close > current close, else 0.
- Features: include at least **RSI** and **MACD** from TA‑Lib, plus any others you like (returns, lags, etc.).

---

## Strategy & Backtest
- At each test bar, if model predicts **up (1)** → hold **long** for the next bar; else **flat**.
- No leverage, no shorting, ignore commissions for simplicity.
- Report at minimum:
  - **Total Return** and **CAGR** (annualized using bar frequency)
  - **Sharpe Ratio** (use 0% risk‑free for simplicity)
  - **Max Drawdown**
  - **Hit Rate** (fraction of correct up/down predictions)

---

## What we look for
- Clean, modular code and reproducibility.
- Sensible feature engineering and evaluation.
- Understanding of **data leakage** and **time‑series validation**.
- Clear README/NOTES with assumptions and how to run.

---

## How to run

1) **Install dependencies** (Python 3.10+ recommended):
```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

> **TA‑Lib** system libs may be required. On macOS (brew): `brew install ta-lib`. On Ubuntu/Debian: `sudo apt-get install -y ta-lib` (or build from source), then `pip install TA-Lib`.

2) **Export your API key**:
```bash
export TWELVEDATA_API_KEY=YOUR_KEY_HERE  # PowerShell: $env:TWELVEDATA_API_KEY="YOUR_KEY_HERE"
```

3) **Run**:
```bash
python -m src.main --ticker AAPL --tf 15m --bars 5000
```

4) **Outputs**:
- Metrics printed to console
- Charts under `artifacts/`
- Model comparison table in console

---

## Extras (optional, nice‑to‑have)
- Walk‑forward validation with `TimeSeriesSplit`
- Basic hyperparameter search
- Transaction cost sensitivity
- Feature importance comparison
- Save artifacts with timestamps
