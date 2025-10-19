"""
Data fetching and processing module for stock prediction.
Handles API calls, data cleaning, feature engineering, and preprocessing.
"""

import os
import requests
import pandas as pd
import numpy as np
import talib as ta
from typing import Tuple, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class DataFetcher:
    """Handles data fetching from Twelve Data API."""
    
    TWELVE_URL = "https://api.twelvedata.com/time_series"
    
    TF_MAP = {
        "1m": "1min",
        "15m": "15min", 
        "1h": "1h",
        "4h": "4h"
    }
    
    def __init__(self):
        self.api_key = self._get_api_key()
    
    def _get_api_key(self) -> str:
        """Get API key from environment variables."""
        key = os.getenv("TWELVEDATA_API_KEY")
        if not key:
            raise RuntimeError("Missing TWELVEDATA_API_KEY environment variable.")
        return key
    
    def fetch_ohlcv(self, ticker: str, timeframe: str, bars: int = 5000) -> pd.DataFrame:
        """
        Fetch OHLCV data from Twelve Data API.
        
        Args:
            ticker: Stock ticker symbol
            timeframe: Timeframe (1m, 15m, 1h, 4h)
            bars: Number of bars to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        if timeframe not in self.TF_MAP:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Choose from {list(self.TF_MAP.keys())}")
        
        params = {
            "symbol": ticker,
            "interval": self.TF_MAP[timeframe],
            "outputsize": bars,
            "apikey": self.api_key,
            "format": "JSON",
            "order": "ASC"
        }
        
        try:
            response = requests.get(self.TWELVE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if "values" not in data:
                if "message" in data:
                    raise RuntimeError(f"API Error: {data['message']}")
                else:
                    raise RuntimeError(f"Unexpected API response: {data}")
            
            if not data["values"]:
                raise RuntimeError(f"No data returned for ticker {ticker}")
            
            df = pd.DataFrame(data["values"])
            
            # Process and clean data
            df = self._process_raw_data(df, ticker)
            return df
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Network error fetching data for {ticker}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error processing data for {ticker}: {e}")
    
    def _process_raw_data(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Process and clean raw API data."""
        # Ensure correct dtypes and order
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Check for missing data
        if df.isnull().any().any():
            print(f"Warning: Missing data detected in {ticker} data")
            df = df.dropna()
        
        if len(df) == 0:
            raise RuntimeError(f"No valid data after cleaning for ticker {ticker}")
        
        df = df.sort_values("datetime").reset_index(drop=True)
        df = df.rename(columns={"datetime": "ts"})
        
        print(f"Successfully fetched {len(df)} bars for {ticker}")
        return df[["ts", "open", "high", "low", "close", "volume"]]


class FeatureEngineer:
    """Handles feature engineering and technical indicators."""
    
    def __init__(self):
        self.feature_columns = [
            "rsi14", "macd", "macdsig", "macdhist", 
            "logret", "vol_ret", "rsi14_lag1", "macd_lag1", "close_lag1"
        ]
    
    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute technical indicators using TA-Lib.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators
        """
        out = df.copy()
        
        # Core technical indicators
        out["rsi14"] = ta.RSI(out["close"].values, timeperiod=14)
        macd, macdsig, macdhist = ta.MACD(out["close"].values, fastperiod=12, slowperiod=26, signalperiod=9)
        out["macd"] = macd
        out["macdsig"] = macdsig
        out["macdhist"] = macdhist
        
        # Returns and volatility
        out["logret"] = np.log(out["close"]).diff()
        out["vol_ret"] = out["logret"].rolling(20).std()
        
        # Lagged features
        out["rsi14_lag1"] = out["rsi14"].shift(1)
        out["macd_lag1"] = out["macd"].shift(1)
        out["close_lag1"] = out["close"].shift(1)
        
        # Drop initial NaNs
        out = out.dropna().reset_index(drop=True)
        return out
    
    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary labels for next-bar direction prediction.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with labels
        """
        out = df.copy()
        out["y"] = (out["close"].shift(-1) > out["close"]).astype(int)
        out = out.iloc[:-1]  # Remove last row (no next close available)
        return out
    
    def get_feature_columns(self) -> list:
        """Get list of feature column names."""
        return self.feature_columns.copy()


class DataSplitter:
    """Handles train/test splitting for time series data."""
    
    @staticmethod
    def time_series_split(df: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically for time series.
        
        Args:
            df: DataFrame to split
            test_ratio: Fraction of data to use for testing
            
        Returns:
            Tuple of (train_df, test_df)
        """
        n = len(df)
        split_idx = int(n * (1 - test_ratio))
        train = df.iloc[:split_idx].copy()
        test = df.iloc[split_idx:].copy()
        return train, test
    
    @staticmethod
    def validate_data_quality(df: pd.DataFrame, min_bars: int = 100, min_samples: int = 50) -> None:
        """
        Validate data quality and raise errors if insufficient.
        
        Args:
            df: DataFrame to validate
            min_bars: Minimum number of bars required
            min_samples: Minimum number of samples after feature engineering
        """
        if len(df) < min_bars:
            raise ValueError(f"Insufficient data: only {len(df)} bars available. Need at least {min_bars} bars for reliable analysis.")
        
        if len(df) < min_samples:
            raise ValueError(f"Insufficient data after feature engineering: only {len(df)} samples available. Need at least {min_samples} samples.")


class DataProcessor:
    """Main data processing pipeline."""
    
    def __init__(self):
        self.fetcher = None  # Will be initialized when needed
        self.engineer = FeatureEngineer()
        self.splitter = DataSplitter()
    
    def process_data(self, ticker: str, timeframe: str, bars: int = 5000) -> Dict[str, Any]:
        """
        Complete data processing pipeline.
        
        Args:
            ticker: Stock ticker symbol
            timeframe: Timeframe
            bars: Number of bars to fetch
            
        Returns:
            Dictionary containing processed data and metadata
        """
        # Initialize fetcher if not already done
        if self.fetcher is None:
            self.fetcher = DataFetcher()
        
        # Fetch data
        print(f"Fetching data for {ticker} @ {timeframe}...")
        df = self.fetcher.fetch_ohlcv(ticker, timeframe, bars)
        print(f"Fetched {len(df)} bars. Computing indicators...")
        
        # Validate initial data
        self.splitter.validate_data_quality(df, min_bars=100)
        
        # Feature engineering
        df_features = self.engineer.compute_indicators(df)
        df_labels = self.engineer.create_labels(df_features)
        
        # Validate after feature engineering
        self.splitter.validate_data_quality(df_labels, min_bars=50, min_samples=50)
        
        # Train/test split
        train_df, test_df = self.splitter.time_series_split(df_labels, test_ratio=0.2)
        
        # Prepare feature matrices
        feature_cols = self.engineer.get_feature_columns()
        X_train = train_df[feature_cols].values
        y_train = train_df["y"].values
        X_test = test_df[feature_cols].values
        y_test = test_df["y"].values
        
        return {
            "train_df": train_df,
            "test_df": test_df,
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "feature_columns": feature_cols,
            "ticker": ticker,
            "timeframe": timeframe
        }
