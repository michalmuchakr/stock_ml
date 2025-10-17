# Design Choices & Tradeoffs

## Data & Labeling
- **Label**: Binary next-bar direction `y_t = 1{close_{t+1} > close_t}` - simple but effective for short-horizon prediction
- **Time Split**: 80% train / 20% test (chronological) - prevents data leakage, critical for time series
- **Features**: RSI(14), MACD(12,26,9), log returns, volatility, and lags - balanced technical analysis approach

## Technical Indicators (TA-Lib)
- **RSI(14)**: Momentum oscillator, good for mean reversion signals
- **MACD(12,26,9)**: Trend-following indicator with signal line
- **Log Returns**: Stationary price changes, better for ML than raw prices
- **Volatility**: 20-period rolling standard deviation of returns
- **Lags**: Previous period values to capture momentum/mean reversion

## Models
- **Random Forest**: Robust, handles non-linear relationships, less prone to overfitting
- **XGBoost**: Gradient boosting, often superior performance, but more prone to overfitting
- **Hyperparameters**: Conservative settings to avoid overfitting (moderate depth, regularization)

## Backtesting Strategy
- **Signal Application**: Model prediction at time `t` applied to next bar `t+1` - realistic execution
- **No Leverage/Shorting**: Conservative approach, long-only strategy
- **No Transaction Costs**: Simplified for this exercise, but important in practice

## Performance Metrics
- **Sharpe Ratio**: Risk-adjusted returns using timeframe-specific annualization
- **Max Drawdown**: Worst peak-to-trough loss - critical risk metric
- **CAGR**: Compound annual growth rate for return comparison
- **Hit Rate**: Accuracy of directional predictions

## Annualization Factors
- 1m: 252 trading days × 390 minutes ≈ 98,280
- 15m: 252 × 26 ≈ 6,552  
- 1h: 252 × 6.5 ≈ 1,638
- 4h: 252 × 1.625 ≈ 409.5

## Tradeoffs & Limitations
- **Short Horizon**: Focus on next-bar prediction, not long-term trends
- **Binary Classification**: Ignores magnitude of moves, only direction
- **No Walk-Forward**: Static train/test split vs. rolling validation
- **Feature Engineering**: Basic technical indicators, could add more sophisticated features
- **Market Regime**: No adaptation to different market conditions

## How to Run
```bash
# Set API key
export TWELVEDATA_API_KEY=your_key_here

# Run with different tickers and timeframes
python -m src.main --ticker AAPL --tf 15m --bars 5000
python -m src.main --ticker MSFT --tf 1h --bars 3000
```

## Implementation Features
- **Robust Error Handling**: Comprehensive validation and error messages
- **Hit Rate Calculation**: Accuracy of directional predictions included in metrics
- **Rolling Sharpe Plots**: Optional rolling Sharpe ratio visualization
- **Enhanced Plotting**: High-quality charts with proper styling and grid
- **Data Validation**: Minimum data requirements and quality checks
- **Modular Design**: Clean separation of concerns between data, models, and visualization

## Output Artifacts
- **Equity curves** for both RandomForest and XGBoost models
- **Rolling Sharpe ratio charts** showing risk-adjusted performance over time
- **Model comparison CSV** with all performance metrics
- **Console output** with detailed results and file locations
- **High-resolution plots** (300 DPI) saved to `artifacts/` directory

## Performance Metrics
- **Total Return**: Cumulative strategy return
- **CAGR**: Compound Annual Growth Rate (annualized)
- **Sharpe Ratio**: Risk-adjusted returns using timeframe-specific annualization
- **Max Drawdown**: Worst peak-to-trough loss
- **Hit Rate**: Fraction of correct directional predictions
- **Accuracy**: Model prediction accuracy on test set
