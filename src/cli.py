"""
Command-line interface module for stock prediction ML pipeline.
Handles argument parsing and CLI configuration.
"""

import argparse


def create_argument_parser():
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Short-horizon stock predictor backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main --ticker AAPL --tf 15m --bars 5000
  python -m src.main --ticker MSFT --tf 1h --bars 3000
  python -m src.main --ticker TSLA --tf 4h --bars 2000
        """
    )
    
    parser.add_argument(
        "--ticker", 
        type=str, 
        required=True, 
        help="US ticker symbol (e.g., AAPL, MSFT, TSLA)"
    )
    
    parser.add_argument(
        "--tf", 
        type=str, 
        required=True, 
        choices=["1m", "15m", "1h", "4h"], 
        help="Timeframe for analysis"
    )
    
    parser.add_argument(
        "--bars", 
        type=int, 
        default=5000, 
        help="Number of bars to fetch (default: 5000)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Output directory for results (default: artifacts)"
    )
    
    parser.add_argument(
        "--tune-hyperparameters",
        action="store_true",
        help="Enable hyperparameter tuning using RandomizedSearchCV and early stopping (faster than grid search)"
    )
    
    return parser
