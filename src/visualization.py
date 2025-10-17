"""
Visualization module for stock prediction results.
Handles plotting of equity curves, performance metrics, and analysis charts.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
from .backtesting import Backtester


class ChartStyler:
    """Handles chart styling and formatting."""
    
    @staticmethod
    def setup_style():
        """Set up matplotlib style for professional charts."""
        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['figure.titlesize'] = 16


class EquityCurvePlotter:
    """Handles plotting of equity curves."""
    
    def __init__(self):
        self.styler = ChartStyler()
        self.styler.setup_style()
    
    def plot_equity_curve(self, backtest_results: pd.DataFrame, title: str, 
                         filename: str, figsize: tuple = (12, 8)) -> None:
        """
        Plot equity curve comparison.
        
        Args:
            backtest_results: DataFrame with asset_ret and strat_ret columns
            title: Chart title
            filename: Output filename
            figsize: Figure size
        """
        # Calculate cumulative returns
        asset_equity = (1 + backtest_results["asset_ret"]).cumprod()
        strategy_equity = (1 + backtest_results["strat_ret"]).cumprod()
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Plot equity curves
        asset_equity.plot(label="Buy & Hold", linewidth=2, color='blue', alpha=0.8)
        strategy_equity.plot(label="Model Strategy", linewidth=2, color='red', alpha=0.8)
        
        # Styling
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.legend(fontsize=12, loc='best')
        plt.xlabel("Time Index", fontsize=12)
        plt.ylabel("Cumulative Return", fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Format y-axis as percentage
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Equity curve saved: {filename}")


class RollingSharpePlotter:
    """Handles plotting of rolling Sharpe ratios."""
    
    def __init__(self, timeframe: str):
        self.backtester = Backtester(timeframe)
        self.styler = ChartStyler()
        self.styler.setup_style()
    
    def plot_rolling_sharpe(self, returns: pd.Series, title: str, filename: str, 
                           window: int = 252, figsize: tuple = (12, 6)) -> None:
        """
        Plot rolling Sharpe ratio.
        
        Args:
            returns: Strategy returns
            title: Chart title
            filename: Output filename
            window: Rolling window size
            figsize: Figure size
        """
        # Calculate rolling Sharpe
        rolling_sharpe = self.backtester.calculate_rolling_sharpe(returns, window)
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Plot rolling Sharpe
        rolling_sharpe.plot(linewidth=2, color='darkblue', alpha=0.8)
        
        # Add zero line
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
        
        # Styling
        plt.title(f"{title} - Rolling Sharpe Ratio ({window} periods)", 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel("Time Index", fontsize=12)
        plt.ylabel("Rolling Sharpe Ratio", fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Add horizontal grid lines
        plt.axhline(y=1, color='green', linestyle=':', alpha=0.5, linewidth=1)
        plt.axhline(y=-1, color='orange', linestyle=':', alpha=0.5, linewidth=1)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Rolling Sharpe chart saved: {filename}")


class PerformanceChartPlotter:
    """Handles plotting of performance comparison charts."""
    
    def __init__(self):
        self.styler = ChartStyler()
        self.styler.setup_style()
    
    def plot_metrics_comparison(self, metrics_df: pd.DataFrame, title: str, 
                               filename: str, figsize: tuple = (12, 8)) -> None:
        """
        Plot performance metrics comparison.
        
        Args:
            metrics_df: DataFrame with performance metrics
            title: Chart title
            filename: Output filename
            figsize: Figure size
        """
        # Select numeric columns for plotting
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
        numeric_df = metrics_df[numeric_cols]
        
        # Create subplots
        n_metrics = len(numeric_cols)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        # Plot each metric
        for i, col in enumerate(numeric_cols):
            ax = axes[i]
            bars = ax.bar(metrics_df['Model'], metrics_df[col], alpha=0.7)
            ax.set_title(f'{col}', fontsize=12, fontweight='bold')
            ax.set_ylabel(col)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Performance comparison chart saved: {filename}")


class VisualizationManager:
    """Main visualization management class."""
    
    def __init__(self, timeframe: str, output_dir: str = "artifacts"):
        self.timeframe = timeframe
        self.output_dir = output_dir
        self.equity_plotter = EquityCurvePlotter()
        self.sharpe_plotter = RollingSharpePlotter(timeframe)
        self.performance_plotter = PerformanceChartPlotter()
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def create_all_plots(self, backtest_results: Dict[str, pd.DataFrame], 
                        performance_metrics: Dict[str, Dict[str, Any]], 
                        ticker: str) -> Dict[str, str]:
        """
        Create all visualization plots.
        
        Args:
            backtest_results: Backtest results for each model
            performance_metrics: Performance metrics for each model
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary of generated filenames
        """
        generated_files = {}
        
        # Create equity curve plots
        for model_name, bt_results in backtest_results.items():
            title = f"Equity Curve {model_name.replace('_', ' ').title()} — {ticker} {self.timeframe}"
            filename = os.path.join(self.output_dir, f"equity_{model_name}_{ticker}_{self.timeframe}.png")
            self.equity_plotter.plot_equity_curve(bt_results, title, filename)
            generated_files[f"equity_{model_name}"] = filename
        
        # Create rolling Sharpe plots
        for model_name, bt_results in backtest_results.items():
            title = f"{model_name.replace('_', ' ').title()} Strategy — {ticker} {self.timeframe}"
            filename = os.path.join(self.output_dir, f"rolling_sharpe_{model_name}_{ticker}_{self.timeframe}.png")
            self.sharpe_plotter.plot_rolling_sharpe(bt_results["strat_ret"], title, filename)
            generated_files[f"rolling_sharpe_{model_name}"] = filename
        
        # Create performance comparison chart
        comparison_df = pd.DataFrame([
            {"Model": name.replace("_", " ").title(), **metrics}
            for name, metrics in performance_metrics.items()
        ])
        
        title = f"Performance Comparison — {ticker} {self.timeframe}"
        filename = os.path.join(self.output_dir, f"performance_comparison_{ticker}_{self.timeframe}.png")
        self.performance_plotter.plot_metrics_comparison(comparison_df, title, filename)
        generated_files["performance_comparison"] = filename
        
        return generated_files
    
    def print_generated_files(self, generated_files: Dict[str, str]) -> None:
        """Print list of generated files."""
        print(f"\nGenerated visualization files in {self.output_dir}/:")
        for file_type, filename in generated_files.items():
            basename = os.path.basename(filename)
            print(f"  - {basename}")
