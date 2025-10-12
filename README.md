# SmartInvest AI

An end-to-end solution for algorithmic trading with AI forecasting and portfolio optimization.

## Overview

SmartInvest AI is a comprehensive platform that combines modern portfolio theory, AI-driven forecasting, and algorithmic trading strategies to help investors make data-driven decisions. The application provides a user-friendly interface built with Streamlit that guides users through the entire investment workflow.

## Features

- **Asset Selection**: Choose from popular stocks or add custom tickers
- **Data Visualization**: Interactive OHLC charts and price history
- **AI Forecasting**: Hybrid LightGBM forecaster with sentiment analysis and regime detection
- **Portfolio Optimization**: 
  - Modern Portfolio Theory (MPT) optimizer
  - Black-Litterman model for incorporating views
- **Backtesting**: Comprehensive backtesting framework with performance metrics
- **Execution**: Paper trading with support for multiple broker adapters:
  - Paper Broker (simulation)
  - Zerodha Kite
  - Alpaca
  - Interactive Brokers

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/SmartInvest-AI.git
cd SmartInvest-AI

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

### Workflow

1. **Select Assets**: Choose tickers in the sidebar
2. **Download Data**: Get historical price data in the Data tab
3. **Generate Forecasts**: Create AI-driven price predictions
4. **Optimize Portfolio**: Apply portfolio optimization techniques
5. **Run Backtest**: Test your strategy with historical data
6. **Execute Trades**: Implement your strategy with paper or live trading

## Requirements

See `requirements.txt` for a complete list of dependencies. Key libraries include:

- streamlit
- pandas
- numpy
- yfinance
- matplotlib
- plotly
- scipy
- scikit-learn
- lightgbm
- hmmlearn
- arch
- transformers
- stable-baselines3
- gym

## Project Structure

- `app.py`: Main Streamlit application
- `models.py`: AI forecasting models
- `portfolio_optimization.py`: Portfolio optimization algorithms
- `Algo_trading.py`: Trading execution adapters
- `backtest.py`: Backtesting framework


## Acknowledgments

- This project was developed for Hackathon Purpose
- Special thanks to [Any acknowledgments]