"""
Main module for stock prediction ML pipeline.
Orchestrates data processing, model training, backtesting, and visualization.
"""

import os
import sys
from typing import Dict, Any

# Import modular components
from .data import DataProcessor
from .models import ModelManager
from .backtesting import BacktestManager
from .visualization import VisualizationManager
from .cli import create_argument_parser


class StockPredictionPipeline:
    """Main pipeline for stock prediction analysis."""
    
    def __init__(self, output_dir: str = "artifacts", enable_hyperparameter_tuning: bool = False):
        self.output_dir = output_dir
        self.data_processor = DataProcessor()
        self.model_manager = ModelManager(enable_hyperparameter_tuning=enable_hyperparameter_tuning)
        self.backtest_manager = None  # Will be initialized with timeframe
        self.visualization_manager = None  # Will be initialized with timeframe
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def run_analysis(self, ticker: str, timeframe: str, bars: int = 5000) -> Dict[str, Any]:
        """
        Run complete stock prediction analysis.
        
        Args:
            ticker: Stock ticker symbol
            timeframe: Timeframe (1m, 15m, 1h, 4h)
            bars: Number of bars to fetch
            
        Returns:
            Dictionary containing all analysis results
        """
        try:
            # Initialize managers with timeframe
            self.backtest_manager = BacktestManager(timeframe)
            self.visualization_manager = VisualizationManager(timeframe, self.output_dir)
            
            print(f"Starting analysis for {ticker} @ {timeframe}")
            print("=" * 50)
            
            # Step 1: Data Processing
            print("\n1. Data Processing")
            print("-" * 20)
            data_results = self.data_processor.process_data(ticker, timeframe, bars)
            
            # Step 2: Model Training and Evaluation
            print("\n2. Model Training and Evaluation")
            print("-" * 30)
            model_results = self.model_manager.train_and_evaluate(
                data_results["X_train"],
                data_results["y_train"],
                data_results["X_test"],
                data_results["y_test"],
                data_results["feature_columns"]
            )
            
            # Step 3: Backtesting
            print("\n3. Backtesting")
            print("-" * 15)
            backtest_results = self.backtest_manager.run_backtests(
                data_results["test_df"]["close"].reset_index(drop=True),
                model_results["predictions"],
                model_results["evaluation_results"]
            )
            
            # Step 4: Visualization
            print("\n4. Visualization")
            print("-" * 15)
            generated_files = self.visualization_manager.create_all_plots(
                backtest_results["backtest_results"],
                backtest_results["performance_metrics"],
                ticker
            )
            
            # Step 5: Save Results
            print("\n5. Saving Results")
            print("-" * 15)
            comparison_table = self.backtest_manager.create_comparison_table(
                backtest_results["performance_metrics"]
            )
            
            csv_filename = os.path.join(self.output_dir, f"model_comparison_{ticker}_{timeframe}.csv")
            self.backtest_manager.save_comparison_table(comparison_table, csv_filename)
            generated_files["comparison_csv"] = csv_filename
            
            # Print summary
            self._print_analysis_summary(comparison_table, generated_files)
            
        except Exception as e:
            print(f"\nError during analysis: {e}")
            print("Please check your API key, ticker symbol, and internet connection.")
            raise
    
    def _print_analysis_summary(self, comparison_table, generated_files):
        """Print analysis summary."""
        print("\n" + "=" * 50)
        print("ANALYSIS COMPLETE")
        print("=" * 50)
        
        print("\nModel Comparison:")
        print(comparison_table.to_string(index=False))
        
        print(f"\nGenerated files in {self.output_dir}/:")
        for file_type, filename in generated_files.items():
            basename = os.path.basename(filename)
            print(f"  - {basename}")


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # Create and run pipeline
        pipeline = StockPredictionPipeline(
            output_dir=args.output_dir,
            enable_hyperparameter_tuning=args.tune_hyperparameters
        )
        pipeline.run_analysis(args.ticker, args.tf, args.bars)
        
        print(f"\nAnalysis completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nAnalysis failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())