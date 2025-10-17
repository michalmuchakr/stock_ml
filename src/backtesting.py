"""
Backtesting module for stock prediction strategies.
Handles strategy simulation, performance metrics, and risk analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple


class Backtester:
    """Handles backtesting of trading strategies."""
    
    ANNUALIZATION_FACTORS = {
        "1m": 252 * 390,
        "15m": 252 * 26,
        "1h": int(252 * 6.5),
        "4h": int(252 * 1.625)
    }
    
    def __init__(self, timeframe: str):
        self.timeframe = timeframe
        self.annualization_factor = self.ANNUALIZATION_FACTORS.get(timeframe, 252)
    
    def backtest_strategy(self, prices: pd.Series, signals: pd.Series) -> pd.DataFrame:
        """
        Backtest a trading strategy.
        
        Args:
            prices: Price series
            signals: Trading signals (1 for long, 0 for flat)
            
        Returns:
            DataFrame with strategy and asset returns
        """
        # Calculate next-bar returns
        asset_returns = prices.pct_change().shift(-1)
        
        # Strategy returns: signal * next_bar_return
        strategy_returns = (signals * asset_returns).fillna(0.0)
        
        return pd.DataFrame({
            "asset_ret": asset_returns.fillna(0.0),
            "strat_ret": strategy_returns.fillna(0.0)
        })
    
    def calculate_performance_metrics(self, returns: pd.Series, hit_rate: float = None) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            returns: Strategy returns
            hit_rate: Hit rate (optional)
            
        Returns:
            Dictionary of performance metrics
        """
        # Basic statistics
        mean_return = returns.mean()
        std_return = returns.std(ddof=0)
        
        # Sharpe ratio
        sharpe_ratio = (mean_return / std_return) * np.sqrt(self.annualization_factor) if std_return > 0 else 0.0
        
        # Drawdown analysis
        equity_curve = (1 + returns).cumprod()
        running_max = equity_curve.cummax()
        drawdown = (equity_curve / running_max) - 1.0
        max_drawdown = drawdown.min()
        
        # Return metrics
        total_return = equity_curve.iloc[-1] - 1.0
        cagr = (1 + mean_return) ** self.annualization_factor - 1 if not np.isnan(mean_return) else np.nan
        
        metrics = {
            "Total Return": float(total_return),
            "CAGR": float(cagr),
            "Sharpe": float(sharpe_ratio),
            "Max Drawdown": float(max_drawdown),
        }
        
        if hit_rate is not None:
            metrics["Hit Rate"] = float(hit_rate)
        
        return metrics
    
    def calculate_rolling_sharpe(self, returns: pd.Series, window: int = 252) -> pd.Series:
        """
        Calculate rolling Sharpe ratio.
        
        Args:
            returns: Strategy returns
            window: Rolling window size
            
        Returns:
            Series of rolling Sharpe ratios
        """
        rolling_mean = returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(self.annualization_factor)
        return rolling_sharpe.fillna(0.0)


class PerformanceAnalyzer:
    """Analyzes and compares strategy performance."""
    
    @staticmethod
    def create_comparison_table(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Create a comparison table of model results.
        
        Args:
            results: Dictionary of model results
            
        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []
        
        for model_name, metrics in results.items():
            row = {"Model": model_name.replace("_", " ").title()}
            row.update(metrics)
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    @staticmethod
    def print_performance_summary(results: Dict[str, Dict[str, Any]]) -> None:
        """
        Print a formatted performance summary.
        
        Args:
            results: Dictionary of model results
        """
        print("\n=== Backtest Performance Summary ===")
        
        for model_name, metrics in results.items():
            print(f"\n{model_name.replace('_', ' ').title()}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
    
    @staticmethod
    def save_results_to_csv(results: pd.DataFrame, filename: str) -> None:
        """
        Save results to CSV file.
        
        Args:
            results: DataFrame with results
            filename: Output filename
        """
        results.to_csv(filename, index=False)
        print(f"Results saved to {filename}")


class BacktestManager:
    """Main backtesting management class."""
    
    def __init__(self, timeframe: str):
        self.backtester = Backtester(timeframe)
        self.analyzer = PerformanceAnalyzer()
        self.timeframe = timeframe
    
    def run_backtests(self, test_prices: pd.Series, predictions: Dict[str, np.ndarray], 
                     evaluation_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Run backtests for all models.
        
        Args:
            test_prices: Test period prices
            predictions: Model predictions
            evaluation_results: Model evaluation results
            
        Returns:
            Dictionary containing backtest results
        """
        backtest_results = {}
        performance_metrics = {}
        
        print("\n=== Running Backtests ===")
        
        for model_name, preds in predictions.items():
            print(f"Backtesting {model_name}...")
            
            # Run backtest
            bt_results = self.backtester.backtest_strategy(test_prices, pd.Series(preds))
            backtest_results[model_name] = bt_results
            
            # Calculate performance metrics
            hit_rate = evaluation_results.get(model_name, {}).get("hit_rate")
            metrics = self.backtester.calculate_performance_metrics(
                bt_results["strat_ret"], hit_rate
            )
            
            # Add accuracy from evaluation
            accuracy = evaluation_results.get(model_name, {}).get("accuracy", 0.0)
            metrics["Accuracy"] = accuracy
            
            performance_metrics[model_name] = metrics
        
        # Print performance summary
        self.analyzer.print_performance_summary(performance_metrics)
        
        return {
            "backtest_results": backtest_results,
            "performance_metrics": performance_metrics
        }
    
    def create_comparison_table(self, performance_metrics: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Create and return comparison table."""
        return self.analyzer.create_comparison_table(performance_metrics)
    
    def save_comparison_table(self, comparison_table: pd.DataFrame, filename: str) -> None:
        """Save comparison table to CSV."""
        self.analyzer.save_results_to_csv(comparison_table, filename)
