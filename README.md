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
- Environment configuration with **`TWELVEDATA_API_KEY`** (see `ENV_SETUP.md` for details).

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

2) **Set up environment variables**:

**Option A: Automated setup (Recommended)**
```bash
python setup_env.py
```

**Option B: Manual setup**
```bash
# Copy the template and edit it
cp env_template.txt .env
# Edit .env and add your API key
export TWELVEDATA_API_KEY=YOUR_KEY_HERE  # PowerShell: $env:TWELVEDATA_API_KEY="YOUR_KEY_HERE"
```

> See [ENV_SETUP.md](ENV_SETUP.md) for detailed environment configuration options.

3) **Run**:
```bash
# Basic run
python -m src.main --ticker AAPL --tf 15m --bars 5000

# With hyperparameter tuning (slower but potentially better performance)
python -m src.main --ticker AAPL --tf 15m --bars 5000 --tune-hyperparameters

# Custom output directory
python -m src.main --ticker MSFT --tf 1h --bars 3000 --output-dir my_results

# Different timeframes
python -m src.main --ticker TSLA --tf 4h --bars 2000
```

4) **Outputs**:
- Metrics printed to console
- Charts under `artifacts/` (or custom output directory)
- Model comparison table in console
- Hyperparameter tuning results (when enabled)
- High-resolution plots (300 DPI) with professional styling

---

## Features

### Hyperparameter Tuning
The pipeline now includes optional hyperparameter tuning using:
- **Random Forest**: RandomizedSearchCV with TimeSeriesSplit for time series validation (faster than GridSearchCV)
- **XGBoost**: BayesSearchCV for intelligent parameter search using Bayesian optimization
- **Parameters tuned**:
  - Random Forest: `n_estimators`, `max_depth`, `min_samples_leaf`, `min_samples_split`, `max_features`
  - XGBoost: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `gamma` (with Bayesian optimization)
- **Time Series Validation**: Proper time series cross-validation to prevent data leakage
- **Efficient Search**: RandomizedSearchCV for Random Forest and BayesSearchCV for XGBoost provide intelligent parameter search

Enable with the `--tune-hyperparameters` flag. This approach balances performance optimization with computational efficiency.

### Command Line Interface
The CLI supports comprehensive options:
- `--ticker`: US stock ticker symbol (required)
- `--tf`: Timeframe - 1m, 15m, 1h, 4h (required)
- `--bars`: Number of bars to fetch (default: 5000)
- `--output-dir`: Output directory for results (default: artifacts)
- `--tune-hyperparameters`: Enable hyperparameter tuning (optional)

### Modular Architecture
The codebase follows a clean modular design:
- **Data Module**: API integration, feature engineering, and preprocessing
- **Models Module**: Machine learning models with hyperparameter tuning
- **Backtesting Module**: Strategy simulation and performance analysis
- **Visualization Module**: Professional charts and plots
- **CLI Module**: Command-line interface and argument parsing
- **Main Module**: Pipeline orchestration and coordination

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed documentation of the modular design.

## Extras (optional, nice‑to‑have)
- Walk‑forward validation with `TimeSeriesSplit` ✅
- Advanced hyperparameter search with RandomizedSearchCV and BayesSearchCV ✅
- Professional visualization with high-resolution charts ✅
- Comprehensive CLI interface ✅
- Time series cross-validation ✅
- Performance comparison charts ✅
- Transaction cost sensitivity
- Feature importance comparison
- Save artifacts with timestamps
