# src/utils.py
import os
import math
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil import tz
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

TWELVE_URL = "https://api.twelvedata.com/time_series"

TF_MAP = {
    "1m": "1min",
    "15m": "15min",
    "1h": "1h",
    "4h": "4h"
}

ANNUALIZATION = {
    "1m": 252*390,
    "15m": 252*26,
    "1h": int(252*6.5),
    "4h": int(252*1.625)
}

def get_api_key() -> str:
    key = os.getenv("TWELVEDATA_API_KEY")
    if not key:
        raise RuntimeError("Missing TWELVEDATA_API_KEY environment variable.")
    return key

def fetch_ohlcv(ticker: str, tf: str, bars: int = 5000) -> pd.DataFrame:
    if tf not in TF_MAP:
        raise ValueError(f"Unsupported timeframe: {tf}. Choose from {list(TF_MAP.keys())}")
    
    params = {
        "symbol": ticker,
        "interval": TF_MAP[tf],
        "outputsize": bars,
        "apikey": get_api_key(),
        "format": "JSON",
        "order": "ASC"
    }
    
    try:
        r = requests.get(TWELVE_URL, params=params, timeout=30)
        r.raise_for_status()
        js = r.json()
        
        if "values" not in js:
            if "message" in js:
                raise RuntimeError(f"API Error: {js['message']}")
            else:
                raise RuntimeError(f"Unexpected API response: {js}")
        
        if not js["values"]:
            raise RuntimeError(f"No data returned for ticker {ticker}")
            
        df = pd.DataFrame(js["values"])
        
        # Ensure correct dtypes and order
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        
        # Check for missing data
        if df.isnull().any().any():
            print(f"Warning: Missing data detected in {ticker} data")
            df = df.dropna()
        
        if len(df) == 0:
            raise RuntimeError(f"No valid data after cleaning for ticker {ticker}")
            
        df = df.sort_values("datetime").reset_index(drop=True)
        df = df.rename(columns={"datetime": "ts"})
        
        print(f"Successfully fetched {len(df)} bars for {ticker}")
        return df[["ts","open","high","low","close","volume"]]
        
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error fetching data for {ticker}: {e}")
    except Exception as e:
        raise RuntimeError(f"Error processing data for {ticker}: {e}")

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    import talib as ta
    out = df.copy()
    # Core indicators
    out["rsi14"] = ta.RSI(out["close"].values, timeperiod=14)
    macd, macdsig, macdhist = ta.MACD(out["close"].values, fastperiod=12, slowperiod=26, signalperiod=9)
    out["macd"] = macd
    out["macdsig"] = macdsig
    out["macdhist"] = macdhist
    # Returns & lags
    out["logret"] = np.log(out["close"]).diff()
    out["vol_ret"] = out["logret"].rolling(20).std()
    out["rsi14_lag1"] = out["rsi14"].shift(1)
    out["macd_lag1"] = out["macd"].shift(1)
    out["close_lag1"] = out["close"].shift(1)
    # Drop initial NaNs
    out = out.dropna().reset_index(drop=True)
    return out

def make_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["y"] = (out["close"].shift(-1) > out["close"]).astype(int)
    out = out.iloc[:-1]  # last row has unknown next close
    return out

def timesplit(df: pd.DataFrame, test_ratio: float = 0.2):
    n = len(df)
    split = int(n*(1-test_ratio))
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()
    return train, test

def perf_metrics(returns: pd.Series, tf: str, hit_rate: float = None) -> dict:
    ann_factor = ANNUALIZATION[tf]
    mean = returns.mean()
    std = returns.std(ddof=0)
    sharpe = (mean / std) * np.sqrt(ann_factor) if std > 0 else 0.0

    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    dd = (equity/peak - 1.0)
    max_dd = dd.min()

    total_return = equity.iloc[-1] - 1.0
    # Approximate CAGR from per-bar compounding
    cagr = (1 + mean)**ann_factor - 1 if not np.isnan(mean) else np.nan

    metrics = {
        "Total Return": float(total_return),
        "CAGR": float(cagr),
        "Sharpe": float(sharpe),
        "Max Drawdown": float(max_dd),
    }
    
    if hit_rate is not None:
        metrics["Hit Rate"] = float(hit_rate)
    
    return metrics

def backtest_signals(prices: pd.Series, signals: pd.Series) -> pd.DataFrame:
    """
    Enter long for next bar if signal_t == 1, else stay flat.
    Strategy return at t+1 = signal_t * (close_{t+1}/close_t - 1).
    """
    rets = prices.pct_change().shift(-1)  # next-bar return aligned with t
    strat = (signals * rets).fillna(0.0)
    return pd.DataFrame({
        "asset_ret": rets.fillna(0.0),
        "strat_ret": strat.fillna(0.0)
    })

def rolling_sharpe(returns: pd.Series, window: int = 252, tf: str = "1h") -> pd.Series:
    """Calculate rolling Sharpe ratio with specified window."""
    ann_factor = ANNUALIZATION[tf]
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(ann_factor)
    return rolling_sharpe.fillna(0.0)
