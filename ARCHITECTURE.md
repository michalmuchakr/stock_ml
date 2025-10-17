# Modular Architecture Documentation

## Overview

The stock prediction ML pipeline has been refactored into a highly modular architecture that promotes code reusability, maintainability, and testability. Each module has a single responsibility and can be used independently or as part of the complete pipeline.

## Module Structure

```
src/
├── __init__.py          # Package initialization and exports
├── main.py              # Main pipeline orchestration
├── data.py              # Data fetching and preprocessing
├── models.py            # Machine learning models and training
├── backtesting.py       # Strategy backtesting and performance metrics
├── visualization.py     # Charts and plots
└── utils.py            # Legacy utilities (deprecated)
```

## Module Descriptions

### 1. Data Module (`data.py`)

**Purpose**: Handles all data-related operations including fetching, cleaning, feature engineering, and preprocessing.

**Key Classes**:
- `DataFetcher`: API integration with Twelve Data
- `FeatureEngineer`: Technical indicators and feature creation
- `DataSplitter`: Time series train/test splitting
- `DataProcessor`: Main data processing pipeline

**Features**:
- Robust error handling for API calls
- Comprehensive feature engineering with TA-Lib
- Proper time series data validation
- Lazy initialization to avoid API key requirements during import

### 2. Models Module (`models.py`)

**Purpose**: Manages machine learning models, training, and evaluation.

**Key Classes**:
- `ModelTrainer`: Model creation and training
- `ModelEvaluator`: Performance evaluation and metrics
- `ModelManager`: Complete model management pipeline

**Features**:
- Support for multiple model types (RandomForest, XGBoost)
- Comprehensive evaluation metrics
- Feature importance analysis
- Configurable hyperparameters

### 3. Backtesting Module (`backtesting.py`)

**Purpose**: Handles strategy backtesting and performance analysis.

**Key Classes**:
- `Backtester`: Individual strategy backtesting
- `PerformanceAnalyzer`: Performance metrics and comparison
- `BacktestManager`: Complete backtesting pipeline

**Features**:
- Realistic strategy simulation
- Comprehensive performance metrics (Sharpe, CAGR, Max Drawdown)
- Rolling Sharpe ratio calculation
- Model comparison functionality

### 4. Visualization Module (`visualization.py`)

**Purpose**: Creates charts, plots, and visualizations.

**Key Classes**:
- `EquityCurvePlotter`: Equity curve visualization
- `RollingSharpePlotter`: Rolling Sharpe ratio charts
- `PerformanceChartPlotter`: Performance comparison charts
- `VisualizationManager`: Complete visualization pipeline

**Features**:
- Professional chart styling
- High-resolution output (300 DPI)
- Multiple chart types
- Automated file management

### 5. Main Module (`main.py`)

**Purpose**: Orchestrates the complete pipeline and provides CLI interface.

**Key Classes**:
- `StockPredictionPipeline`: Main pipeline coordinator

**Features**:
- Command-line interface with comprehensive help
- Step-by-step progress reporting
- Error handling and user feedback
- Configurable output directory

## Benefits of Modular Architecture

### 1. **Separation of Concerns**
Each module has a single, well-defined responsibility, making the code easier to understand and maintain.

### 2. **Reusability**
Individual modules can be imported and used independently for specific tasks.

### 3. **Testability**
Each module can be unit tested in isolation, making it easier to identify and fix issues.

### 4. **Maintainability**
Changes to one module don't affect others, reducing the risk of introducing bugs.

### 5. **Extensibility**
New features can be added by creating new modules or extending existing ones.

### 6. **Flexibility**
The pipeline can be customized by replacing or extending individual modules.

## Usage Examples

### Using Individual Modules

```python
from src.data import FeatureEngineer, DataSplitter
from src.models import ModelManager
from src.backtesting import BacktestManager

# Use individual components
engineer = FeatureEngineer()
features = engineer.compute_indicators(df)

model_manager = ModelManager()
models = model_manager.train_models(X_train, y_train)

backtest_manager = BacktestManager("15m")
results = backtest_manager.run_backtests(prices, predictions, metrics)
```

### Using the Complete Pipeline

```python
from src.main import StockPredictionPipeline

# Run complete analysis
pipeline = StockPredictionPipeline("output_dir")
results = pipeline.run_analysis("AAPL", "15m", 5000)
```

### Command Line Usage

```bash
python -m src.main --ticker AAPL --tf 15m --bars 5000 --output-dir results
```

## Testing

The modular architecture includes comprehensive testing:

- **Unit Tests**: Each module can be tested independently
- **Integration Tests**: Full pipeline testing with synthetic data
- **Error Handling**: Robust error handling and validation
- **Mock Support**: Easy mocking for testing without API dependencies

## Migration from Legacy Code

The original `utils.py` file has been deprecated in favor of the modular structure. All functionality has been preserved and enhanced:

- Data fetching → `DataFetcher` class
- Feature engineering → `FeatureEngineer` class
- Model training → `ModelTrainer` class
- Backtesting → `Backtester` class
- Visualization → Various plotter classes

## Future Enhancements

The modular architecture makes it easy to add new features:

1. **New Data Sources**: Extend `DataFetcher` or create new fetcher classes
2. **Additional Models**: Add new model types to `ModelTrainer`
3. **Advanced Metrics**: Extend `PerformanceAnalyzer` with new metrics
4. **Custom Visualizations**: Add new plotter classes
5. **Real-time Processing**: Create streaming data processors

## Best Practices

1. **Single Responsibility**: Each class has one clear purpose
2. **Dependency Injection**: Dependencies are injected rather than hardcoded
3. **Error Handling**: Comprehensive error handling at each level
4. **Documentation**: Clear docstrings and type hints
5. **Testing**: Each module is thoroughly tested
6. **Configuration**: Configurable parameters and settings
