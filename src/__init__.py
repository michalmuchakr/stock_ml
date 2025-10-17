"""
Stock Prediction ML Package

A modular machine learning pipeline for short-horizon stock movement prediction.
Includes data fetching, feature engineering, model training, backtesting, and visualization.

Modules:
- data: Data fetching and preprocessing
- models: Machine learning models and training
- backtesting: Strategy backtesting and performance metrics
- visualization: Charts and plots
- main: Main pipeline orchestration
"""

from .data import DataProcessor, DataFetcher, FeatureEngineer, DataSplitter
from .models import ModelManager, ModelTrainer, ModelEvaluator
from .backtesting import BacktestManager, Backtester, PerformanceAnalyzer
from .visualization import VisualizationManager, EquityCurvePlotter, RollingSharpePlotter
from .main import StockPredictionPipeline

__version__ = "1.0.0"
__author__ = "Stock ML Team"

__all__ = [
    # Main pipeline
    "StockPredictionPipeline",
    
    # Data modules
    "DataProcessor",
    "DataFetcher", 
    "FeatureEngineer",
    "DataSplitter",
    
    # Model modules
    "ModelManager",
    "ModelTrainer",
    "ModelEvaluator",
    
    # Backtesting modules
    "BacktestManager",
    "Backtester",
    "PerformanceAnalyzer",
    
    # Visualization modules
    "VisualizationManager",
    "EquityCurvePlotter",
    "RollingSharpePlotter",
]
