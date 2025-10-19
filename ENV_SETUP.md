# Environment Setup Guide

This guide explains how to set up environment variables for the Stock ML project.

## Quick Setup

### Option 1: Automated Setup (Recommended)
```bash
python setup_env.py
```
This script will:
- Create a `.env` file from the template
- Prompt you for your Twelve Data API key
- Set up all necessary environment variables

### Option 2: Manual Setup
1. Copy the template file:
   ```bash
   cp env_template.txt .env
   ```

2. Edit the `.env` file and replace `your_twelvedata_api_key_here` with your actual API key.

## Required Environment Variables

### TWELVEDATA_API_KEY (REQUIRED)
- **Purpose**: API key for fetching stock data from Twelve Data
- **How to get**: Sign up at [twelvedata.com](https://twelvedata.com/) for a free API key
- **Example**: `TWELVEDATA_API_KEY=abc123def456ghi789`

## Optional Environment Variables

The `.env` file includes many optional configuration variables organized by category:

### Application Configuration
- `OUTPUT_DIR`: Directory for output files (default: `artifacts`)
- `DEFAULT_BARS`: Number of bars to fetch (default: `5000`)
- `TEST_RATIO`: Train/test split ratio (default: `0.2`)
- `RANDOM_STATE`: Random seed for reproducibility (default: `42`)

### Model Configuration
- `RF_N_ESTIMATORS`: Random Forest number of estimators (default: `300`)
- `RF_MAX_DEPTH`: Random Forest max depth (default: `6`)
- `XGB_N_ESTIMATORS`: XGBoost number of estimators (default: `400`)
- `XGB_LEARNING_RATE`: XGBoost learning rate (default: `0.05`)

### Backtesting Configuration
- `PREDICTION_THRESHOLD`: Classification threshold (default: `0.5`)
- `ROLLING_SHARPE_WINDOW`: Rolling Sharpe window size (default: `252`)

### Visualization Configuration
- `CHART_DPI`: Chart resolution (default: `300`)
- `CHART_WIDTH`: Chart width (default: `12`)
- `CHART_HEIGHT`: Chart height (default: `8`)

### Development Configuration
- `DEBUG`: Enable debug mode (default: `false`)
- `VERBOSE`: Enable verbose output (default: `true`)
- `VALIDATE_DATA`: Enable data validation (default: `true`)

## Usage

Once your `.env` file is set up, you can run the stock prediction pipeline:

```bash
# Basic usage
python -m src.main --ticker AAPL --tf 15m --bars 5000

# With custom output directory
python -m src.main --ticker MSFT --tf 1h --bars 3000 --output-dir results

# Different timeframe
python -m src.main --ticker TSLA --tf 4h --bars 2000
```

## Environment Variable Loading

The project automatically loads environment variables from the `.env` file. The current implementation uses:

```python
import os
api_key = os.getenv("TWELVEDATA_API_KEY")
```

## Security Notes

- **Never commit your `.env` file** - it's already in `.gitignore`
- **Keep your API key secure** - don't share it publicly
- **Use the template file** (`env_template.txt`) for version control
- **Rotate your API keys** periodically for security

## Troubleshooting

### Common Issues

1. **"Missing TWELVEDATA_API_KEY environment variable"**
   - Make sure you've created a `.env` file
   - Verify the API key is correctly set in the file
   - Check for typos in the variable name

2. **API key not working**
   - Verify your API key is valid at [twelvedata.com](https://twelvedata.com/)
   - Check if you have sufficient API quota
   - Ensure you're using the correct API key format

3. **Environment variables not loading**
   - Make sure the `.env` file is in the project root directory
   - Check that the file format is correct (no spaces around `=`)
   - Restart your terminal/IDE after creating the file

### File Structure
```
stock_ml/
├── .env                    # Your environment variables (DO NOT COMMIT)
├── env_template.txt        # Template file (safe to commit)
├── setup_env.py           # Setup script
├── ENV_SETUP.md           # This guide
└── src/
    └── ...
```

## Support

If you encounter issues with environment setup:
1. Check this guide first
2. Verify your API key is valid
3. Ensure all required dependencies are installed
4. Check the main README.md for additional setup instructions
