#!/usr/bin/env python3
"""
Comprehensive test script for the modular stock prediction pipeline.
Tests all components with synthetic data to ensure everything works correctly.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add src to path
sys.path.append('src')

from src.data import DataProcessor, DataFetcher, FeatureEngineer, DataSplitter
from src.models import ModelManager, ModelTrainer, ModelEvaluator
from src.backtesting import BacktestManager, Backtester, PerformanceAnalyzer
from src.visualization import VisualizationManager, EquityCurvePlotter, RollingSharpePlotter, PerformanceChartPlotter
from src.main import StockPredictionPipeline


def generate_synthetic_data(n_bars=1000):
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    
    # Generate random walk for prices
    returns = np.random.normal(0.0001, 0.02, n_bars)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    data = []
    for i, close in enumerate(prices):
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_price = close * (1 + np.random.normal(0, 0.005))
        volume = np.random.randint(1000000, 10000000)
        
        data.append({
            'ts': pd.Timestamp.now() + pd.Timedelta(minutes=15*i),
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data)


def test_data_module():
    """Test data processing module."""
    print("Testing Data Module...")
    print("-" * 30)
    
    # Test feature engineering
    engineer = FeatureEngineer()
    df = generate_synthetic_data(500)
    
    # Test indicators
    df_features = engineer.compute_indicators(df)
    print(f"✓ Indicators computed: {len(df_features)} samples")
    
    # Test labeling
    df_labels = engineer.create_labels(df_features)
    print(f"✓ Labels created: {len(df_labels)} samples")
    
    # Test data splitting
    splitter = DataSplitter()
    train, test = splitter.time_series_split(df_labels, test_ratio=0.2)
    print(f"✓ Data split: {len(train)} train, {len(test)} test")
    
    # Test validation
    splitter.validate_data_quality(df_labels, min_bars=50, min_samples=50)
    print("✓ Data validation passed")
    
    return df_labels, train, test


def test_models_module(train_df, test_df):
    """Test models module."""
    print("\nTesting Models Module...")
    print("-" * 30)
    
    # Prepare data
    engineer = FeatureEngineer()
    feature_cols = engineer.get_feature_columns()
    X_train = train_df[feature_cols].values
    y_train = train_df["y"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["y"].values
    
    # Test model manager
    model_manager = ModelManager()
    results = model_manager.train_and_evaluate(X_train, y_train, X_test, y_test, feature_cols)
    
    print(f"✓ Models trained: {len(results['models'])} models")
    print(f"✓ Predictions made: {len(results['predictions'])} models")
    print(f"✓ Evaluation completed: {len(results['evaluation_results'])} models")
    
    # Test individual components
    trainer = ModelTrainer()
    models = trainer.create_models()
    print(f"✓ Model creation: {len(models)} models")
    
    evaluator = ModelEvaluator()
    accuracy = evaluator.calculate_accuracy(y_test, results['predictions']['random_forest'])
    hit_rate = evaluator.calculate_hit_rate(y_test, results['predictions']['random_forest'])
    print(f"✓ Sample metrics - Accuracy: {accuracy:.4f}, Hit Rate: {hit_rate:.4f}")
    
    return results


def test_backtesting_module(test_df, model_results):
    """Test backtesting module."""
    print("\nTesting Backtesting Module...")
    print("-" * 30)
    
    # Test backtester
    backtester = Backtester("15m")
    test_prices = test_df["close"].reset_index(drop=True)
    
    # Test individual backtest
    bt_result = backtester.backtest_strategy(test_prices, pd.Series(model_results['predictions']['random_forest']))
    print(f"✓ Individual backtest: {len(bt_result)} periods")
    
    # Test performance metrics
    metrics = backtester.calculate_performance_metrics(bt_result["strat_ret"], 0.5)
    print(f"✓ Performance metrics: {len(metrics)} metrics")
    
    # Test rolling Sharpe
    rolling_sharpe = backtester.calculate_rolling_sharpe(bt_result["strat_ret"], window=50)
    print(f"✓ Rolling Sharpe: {len(rolling_sharpe)} periods")
    
    # Test backtest manager
    backtest_manager = BacktestManager("15m")
    backtest_results = backtest_manager.run_backtests(
        test_prices, 
        model_results['predictions'], 
        model_results['evaluation_results']
    )
    print(f"✓ Backtest manager: {len(backtest_results['backtest_results'])} models")
    
    # Test comparison table
    comparison_table = backtest_manager.create_comparison_table(backtest_results['performance_metrics'])
    print(f"✓ Comparison table: {len(comparison_table)} rows")
    
    return backtest_results


def test_visualization_module(backtest_results, ticker="TEST"):
    """Test visualization module."""
    print("\nTesting Visualization Module...")
    print("-" * 30)
    
    # Test individual plotters
    equity_plotter = EquityCurvePlotter()
    sharpe_plotter = RollingSharpePlotter("15m")
    performance_plotter = PerformanceChartPlotter()
    
    # Test equity curve plotting
    test_bt = backtest_results['backtest_results']['random_forest']
    equity_plotter.plot_equity_curve(test_bt, "Test Equity Curve", "test_equity.png")
    print("✓ Equity curve plotting")
    
    # Test rolling Sharpe plotting
    sharpe_plotter.plot_rolling_sharpe(test_bt["strat_ret"], "Test Rolling Sharpe", "test_sharpe.png")
    print("✓ Rolling Sharpe plotting")
    
    # Test performance comparison
    comparison_df = pd.DataFrame([
        {"Model": "Test Model", **backtest_results['performance_metrics']['random_forest']}
    ])
    performance_plotter.plot_metrics_comparison(comparison_df, "Test Comparison", "test_comparison.png")
    print("✓ Performance comparison plotting")
    
    # Test visualization manager
    viz_manager = VisualizationManager("15m", "test_artifacts")
    generated_files = viz_manager.create_all_plots(
        backtest_results['backtest_results'],
        backtest_results['performance_metrics'],
        ticker
    )
    print(f"✓ Visualization manager: {len(generated_files)} files generated")
    
    return generated_files


def test_full_pipeline():
    """Test the complete pipeline."""
    print("\nTesting Full Pipeline...")
    print("-" * 30)
    
    # Create a mock data processor that uses synthetic data
    class MockDataProcessor:
        def __init__(self):
            self.engineer = FeatureEngineer()
            self.splitter = DataSplitter()
        
        def process_data(self, ticker, timeframe, bars=5000):
            # Generate synthetic data instead of fetching from API
            df = generate_synthetic_data(bars)
            
            # Process the data
            df_features = self.engineer.compute_indicators(df)
            df_labels = self.engineer.create_labels(df_features)
            
            # Validate
            self.splitter.validate_data_quality(df_labels, min_bars=50, min_samples=50)
            
            # Split
            train_df, test_df = self.splitter.time_series_split(df_labels, test_ratio=0.2)
            
            # Prepare features
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
    
    # Create pipeline with mock data processor
    pipeline = StockPredictionPipeline("test_artifacts")
    pipeline.data_processor = MockDataProcessor()
    
    # Run analysis
    results = pipeline.run_analysis("TEST", "15m", 1000)
    
    print(f"✓ Full pipeline completed")
    print(f"✓ Generated {len(results['generated_files'])} files")
    print(f"✓ Comparison table: {len(results['comparison_table'])} rows")
    
    return results


def cleanup_test_files():
    """Clean up test files."""
    test_files = [
        "test_equity.png",
        "test_sharpe.png", 
        "test_comparison.png",
        "test_artifacts"
    ]
    
    for file in test_files:
        if os.path.exists(file):
            if os.path.isdir(file):
                import shutil
                shutil.rmtree(file)
            else:
                os.remove(file)
    print("✓ Test files cleaned up")


def main():
    """Run all tests."""
    print("=" * 60)
    print("COMPREHENSIVE MODULAR PIPELINE TEST")
    print("=" * 60)
    
    try:
        # Test individual modules
        df_labels, train_df, test_df = test_data_module()
        model_results = test_models_module(train_df, test_df)
        backtest_results = test_backtesting_module(test_df, model_results)
        generated_files = test_visualization_module(backtest_results)
        
        # Test full pipeline
        pipeline_results = test_full_pipeline()
        
        # Cleanup
        cleanup_test_files()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nModular architecture is working correctly:")
        print("✓ Data processing module")
        print("✓ Models module") 
        print("✓ Backtesting module")
        print("✓ Visualization module")
        print("✓ Full pipeline integration")
        print("\nThe code is now highly modular and maintainable!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
