import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import base64

# Import custom modules
from models import PipelineCoordinator
from portfolio_optimization import MPTOptimizer, BlackLitterman, backtest_weights
from Algo_trading import SizeConfig, ExecConfig
from backtest import vectorized_backtest_ohlc 

# Set page config
st.set_page_config(
    page_title="SmartInvest AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.markdown("<h1 class='main-header'>SmartInvest AI</h1>", unsafe_allow_html=True)
st.markdown("<div class='info-box'>An end-to-end solution for algorithmic trading with AI forecasting and portfolio optimization.</div>", unsafe_allow_html=True)

# Initialize session_state for tickers if not exists
if 'selected_tickers' not in st.session_state:
    st.session_state['selected_tickers'] = []

# Sidebar for asset selection
with st.sidebar:
    st.markdown("<div class='sidebar-content'>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:#1E88E5;'>Configuration</h2>", unsafe_allow_html=True)
    
    # Asset selection
    st.markdown("<h3 style='color:#0D47A1;'>Asset Selection</h3>", unsafe_allow_html=True)
    default_tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX', 'JPM', 'BAC']
    
    for ticker in default_tickers:
        if st.checkbox(ticker, value=False):
            if ticker not in st.session_state['selected_tickers']:
                st.session_state['selected_tickers'].append(ticker)
        else:
            if ticker in st.session_state['selected_tickers']:
                st.session_state['selected_tickers'].remove(ticker)
    
    # Custom ticker input
    custom_ticker = st.text_input("Add custom ticker:")
    if st.button("Add Ticker") and custom_ticker:
        try:
            # Validate ticker
            ticker_check = yf.Ticker(custom_ticker)
            if hasattr(ticker_check, 'info') and ticker_check.info.get('regularMarketPrice') is not None:
                if custom_ticker not in st.session_state['selected_tickers']:
                    st.session_state['selected_tickers'].append(custom_ticker)
                    st.success(f"Added {custom_ticker}")
                else:
                    st.warning(f"{custom_ticker} already added")
            else:
                st.error(f"Invalid ticker: {custom_ticker}")
        except Exception as e:
            st.error(f"Error validating ticker: {e}")
    
    # Date range selection
    st.markdown("<h3 style='color:#0D47A1;'>Date Range</h3>", unsafe_allow_html=True)
    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365*2))
    end_date = st.date_input("End Date", datetime.now())
    
    if start_date >= end_date:
        st.error("Start date must be before end date")
    
    # Forecasting options
    st.markdown("<h3 style='color:#0D47A1;'>Forecasting Uses</h3>", unsafe_allow_html=True)
    st.write("🔬 Hybrid LGB Forecaster")
    st.write("🧠 Sentiment Analysis")
    st.write("⚡ Regime Detection")

    
    # Optimizer selection
    st.markdown("<h3 style='color:#0D47A1;'>Optimizer</h3>", unsafe_allow_html=True)

    st.write("🔹 MPT Optimizer")
    st.write("🔸 Black-Litterman")

    st.markdown("<h3 style='color:#0D47A1;'>Adapter</h3>", unsafe_allow_html=True)

    st.write("🔹 Paper Broker")

    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Help section
    with st.expander("Help & Information"):
        st.markdown("""
        **How to use this app:**
        1. Select assets in the sidebar
        2. Download data in the Data tab
        3. Generate forecasts in the Forecast tab
        4. Optimize your portfolio in the Optimization tab
        5. Run a backtest in the Backtest tab
        6. View results in the Results tab
        
        For more information, contact support@smartinvestai.com
        """)


if 'selected_tickers' in st.session_state and st.session_state['selected_tickers']:
    st.write(f"Selected tickers: {', '.join(st.session_state['selected_tickers'])}")
else:
    st.warning("Please select at least one ticker to proceed.")

# Create tabs for the main workflow
tabs = st.tabs(["📊 Data", "🤖 Forecast", "🧮 Optimization", "🧪 Backtest", "🛠️ Execution", "📈 Results"])

# Cache data download function
@st.cache_data(ttl=3600)
def download_data(tickers, start_date, end_date):
    data = {}
    for ticker in tickers:
        try:
            ticker_data = yf.download(ticker, start=start_date, end=end_date)
            if not ticker_data.empty:
                data[ticker] = ticker_data
            else:
                st.warning(f"No data available for {ticker} in the selected date range.")
        except Exception as e:
            st.error(f"Error downloading data for {ticker}: {e}")
    
    if not data:
        raise Exception("Failed to download data for any of the selected tickers. Please check your internet connection or try different tickers.")
    
    return data

# Function to create OHLC chart
def plot_ohlc(data, ticker):
    df = data.copy()
    df.columns = df.columns.get_level_values(0)
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name=ticker
    )])
    fig.update_layout(
        title=f"{ticker} OHLC Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        height=500
    )
    return fig

# Function to get download link for CSV
def get_csv_download_link(df, filename, text):
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">{text}</a>'
    return href

# Main workflow
if 'selected_tickers' in st.session_state and st.session_state['selected_tickers']:
    # Data Download (Step 1)
    with tabs[0]:
        st.header("Data Download")

        # Download button
        download_button = st.button("Download Data")
    
        if download_button:
            with st.spinner("Downloading data..."):
                data = download_data(st.session_state["selected_tickers"], start_date, end_date)

                if data:
                    st.session_state['data'] = data  # store in session_state
                    st.success(f"Successfully downloaded data for {len(data)} tickers.")
                else:
                    st.error("Failed to download data. Please try again.")

        # --- Display data if already in session_state ---
        if 'data' in st.session_state and st.session_state['data']:
            data = st.session_state['data']
            for ticker, ticker_data in data.items():
                st.subheader(f"{ticker} Data")
                st.dataframe(ticker_data.tail())
                st.plotly_chart(plot_ohlc(ticker_data, ticker), use_container_width=True)

    # Forecasting Layer (Step 2)
    with tabs[1]:
        st.header("Forecasting")

        if 'data' not in st.session_state:
            st.info("Please download data first in the Data tab.")
        else:
            # Button to generate forecasts
            forecast_button = st.button("Generate Forecasts")

            if forecast_button:
                with st.spinner("Generating forecasts..."):
                    # Prepare price dataframe
                    prices_df = pd.DataFrame() 
                    for ticker, ticker_data in st.session_state['data'].items(): 
                        prices_df[ticker] = ticker_data['Close']
                    
                    # Initialize pipeline
                    pipeline = PipelineCoordinator(tickers=prices_df.columns, use_rl=False)
                    sentiment_df, sentiment_wide = pipeline.sentiment_analyzer.build_dataframe(prices_df.columns)

                    features = pipeline.build_features(prices_df, sentiment_wide)
                    pipeline.fit_return_model(prices_df, macro_df=sentiment_wide)
                    pipeline.fit_regime(features)
                    pipeline.fit_risk(prices_df)
                    results = pipeline.run_pipeline(price_df=prices_df, feature_matrix=features, macro_df=sentiment_wide)
                    results['sentiment'] = {'df': sentiment_df, 'daily_pivot': sentiment_wide}
                    weights = pipeline.generate_signals(features, prices_df, sentiment_wide, top_n=2)

                    # Store forecasts in session_state
                    st.session_state['results'] = results
                    st.session_state['prices_df'] = prices_df
                    st.session_state['weights'] = weights

            # --- Display forecasts if available ---
            if 'results' in st.session_state:
                results = st.session_state['results']
                prices_df = st.session_state['prices_df']

                try:
                    exp_returns = results['forecaster']['expected_returns']
                    exp_cov = results['forecaster']['covariance']
                    selected_tickers = list(prices_df.columns)

                    st.session_state['exp_returns'] = results['forecaster']['expected_returns']
                    st.session_state['exp_cov'] = results['forecaster']['covariance']


                    forecast_df = pd.DataFrame({
                        'Ticker': selected_tickers,
                        'Expected Return (%)': np.round(exp_returns * 100, 2),
                        'Expected Volatility (%)': np.round(np.sqrt(np.diag(exp_cov))*np.sqrt(252), 4)
                    })

                    st.subheader("Return and Volatility Forecasts")
                    st.dataframe(forecast_df)

                    st.subheader("Sentiment Analysis")
                    st.dataframe(results['sentiment']['daily_pivot'].tail())

                    # Risk-Return scatter plot
                    fig = px.scatter(
                        forecast_df,
                        x='Expected Volatility (%)',
                        y='Expected Return (%)',
                        text='Ticker',
                        title='Risk-Return Profile'
                    )
                    fig.update_traces(textposition='top center')
                    st.plotly_chart(fig, use_container_width=True)

                    # Market regime detection
                    if 'regime' in results:
                        st.subheader("Market Regime Detection")
                        fig = go.Figure()
                        for ticker in selected_tickers:
                            fig.add_trace(go.Scatter(
                                x=prices_df.index,
                                y=prices_df[ticker],
                                mode='lines',
                                name=ticker
                            ))
                        fig.update_layout(
                            title="Market Regimes",
                            xaxis_title="Date",
                            yaxis_title="Price",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    


                except Exception as e:
                    st.error(f"Error displaying forecasts: {e}")
        
    # Portfolio Optimization (Step 3)
    with tabs[2]:
        st.header("Portfolio Optimization")

        if 'exp_returns' not in st.session_state or 'exp_cov' not in st.session_state:
            st.info("Please generate forecasts first in the Forecasting tab.")
        else:
            # --- User Inputs ---
            risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 5.0, 2.0) / 100
            max_weight = st.slider("Maximum Weight per Asset (%)", 5, 100, 20)
            allow_shorting = st.checkbox("Allow Shorting", value=False)

            optimize_button = st.button("Optimize Portfolio")

            # Run optimization if button clicked
            if optimize_button:
                with st.spinner("Optimizing portfolio..."):
                    try:
                        exp_rets = st.session_state['exp_returns']
                        exp_cov = st.session_state['exp_cov']
                        # Initialize MPT optimizer
                        mk = np.ones(len(exp_rets)) / len(exp_rets)
                        bl = BlackLitterman(cov=exp_cov, tau=0.05, delta=2.5)
                        pi = bl.equilibrium_returns(mk)
                        mu_bl, sigma_bl = bl.posterior(pi, np.eye(len(exp_rets)), exp_rets, Omega=None)
                        mpt = MPTOptimizer(expected_returns=mu_bl*256, cov=sigma_bl*256,
                                        max_weight=max_weight/100, risk_free=risk_free_rate)

                        # Get optimized portfolios
                        max_sharpe_weights = mpt.max_sharpe()
                        min_vol_weights = mpt.min_variance()

                        # Store results in session_state
                        st.session_state['optimized_weights'] = max_sharpe_weights
                        st.session_state['optimized_min_vol'] = min_vol_weights


                    except Exception as e:
                        st.error(f"Error in optimization: {e}")

            # --- Display results if available ---
            if 'optimized_weights' in st.session_state:
                weights = st.session_state['optimized_weights']
                min_vol_weights = st.session_state.get('optimized_min_vol')
                weights_df = pd.DataFrame({
                    'Ticker': st.session_state['selected_tickers'],
                    'Max Sharpe Weights (%)': np.round(weights * 100, 2),
                    'Min Volatility Weights (%)': np.round(min_vol_weights * 100, 2)
                })
                st.subheader("Optimized Portfolio Weights")
                st.dataframe(weights_df)

                # Plot efficient frontier
                exp_rets = st.session_state['exp_returns']
                exp_cov = st.session_state['exp_cov']
                mpt = MPTOptimizer(exp_rets, exp_cov, max_weight=max_weight/100, risk_free=risk_free_rate)
                frontier = mpt.efficient_frontier()
                ef_returns = [f['target_return'] for f in frontier]
                ef_volatility = [f['volatility'] for f in frontier]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ef_volatility, y=ef_returns, mode='lines+markers', name='Efficient Frontier'))
                fig.update_layout(title="Efficient Frontier", xaxis_title='Volatility', yaxis_title='Expected Return')
                st.plotly_chart(fig, use_container_width=True)

                fig2 = go.Figure()
                
                prices = pd.DataFrame() 
                for ticker, ticker_data in st.session_state['data'].items(): 
                    prices[ticker] = ticker_data['Close']
                w_sharpe = mpt.max_sharpe()
                w_var = mpt.min_variance()

                cum_min, ret_min = backtest_weights(prices, min_vol_weights)
                cum_max, ret_max = backtest_weights(prices, weights) 

                fig2.add_trace(go.Scatter(
                    x=cum_min.index,
                    y=cum_min.values,
                    mode='lines',
                    name='Min Variance'
                ))

                fig2.add_trace(go.Scatter(
                    x=cum_max.index,
                    y=cum_max.values,
                    mode='lines',
                    name='Max Sharpe'
                ))

                fig2.update_layout(
                    title='Buy-and-Hold Backtest (Fixed Weights from Model)',
                    xaxis_title='Date',
                    yaxis_title='Cumulative Returns',
                    legend_title='Portfolio Type',
                    template='plotly_white',
                    height=500,
                )
                st.plotly_chart(fig2, use_container_width=True)

        
    # Backtest & Execution (Step 4)
    with tabs[3]:
        st.header("Backtest & Execution")

        if 'optimized_weights' not in st.session_state:
            st.info("Please optimize portfolio first in the Portfolio Optimization tab.")
        else:
            # --- Backtest Parameters ---
            initial_capital = st.number_input("Initial Capital ($)", min_value=1000, value=100000, step=1000)
            st.session_state['initial_capital'] = initial_capital
            commission = st.slider("Commission (%)", 0.0, 1.0, 0.1) / 100

            # Risk management
            use_stop_loss = st.checkbox("Use Stop Loss", value=True)
            stop_loss_pct = st.slider("Stop Loss (%)", 1.0, 10.0, 5.0) / 100

            use_take_profit = st.checkbox("Use Take Profit", value=True)
            take_profit_pct = st.slider("Take Profit (%)", 1.0, 20.0, 10.0) / 100

            backtest_button = st.button("Run Backtest")

            # Run backtest if button clicked
            if backtest_button:
                with st.spinner("Running backtest..."):
                    try:
                        ohlc_data = {ticker: st.session_state['data'][ticker] for ticker in st.session_state['selected_tickers']}

                        cfg = ExecConfig(
                            transaction_cost=commission,
                            slippage=0.0005,
                            size_config=SizeConfig(
                                mode='percent_of_equity',
                                percent_of_equity=0.1
                            ),
                            stop_loss_pct=stop_loss_pct,
                            take_profit_pct=use_take_profit,
                            max_exposure_per_ticker=15000,
                            max_exposure_per_sector=30000,
                            max_total_leverage=1.0
                        )

                        sector_map = {
                            'AAPL': 'Technology',
                            'MSFT': 'Technology',
                            'META': 'Communication Services',
                            'GOOG': 'Communication Services',
                            'IAU': 'Precious Metals',
                            'GLD': 'Precious Metals',
                            'IEF': 'Fixed Income',
                            'BTC-USD': 'Cryptocurrency'
                        }

                        # Run vectorized backtest
                        backtest_results = vectorized_backtest_ohlc(
                            ohlc_data,
                            st.session_state['optimized_weights'],
                            cfg,
                            sector_map
                        )

                        # Store results in session state
                        st.session_state['backtest_results'] = backtest_results
                        st.success("Strategy execution completed!")

                    except Exception as e:
                        st.error(f"Error in backtest: {e}")

            # --- Display backtest results if available ---
            if 'backtest_results' in st.session_state:
                backtest_results = st.session_state['backtest_results']

                # Equity curve
                st.subheader("Equity Curve")
                fig = px.line(backtest_results['equity_curve'], title='Portfolio Equity Curve')
                st.plotly_chart(fig, use_container_width=True)

                # Trade statistics
                st.subheader("Trade Statistics")
                trade_stats = pd.DataFrame({
                    'Metric': ['Total Trades', 'Win Rate (%)', 'Avg. Profit (%)', 'Avg. Loss (%)', 'Profit Factor'],
                    'Value': [
                        backtest_results['trade_stats']['total_closed_trades'],
                        round(backtest_results['trade_stats']['win_rate'] * 100, 2),
                        round(backtest_results['trade_stats']['avg_profit_amt'] * 100, 2),
                        round(backtest_results['trade_stats']['avg_loss_amt'] * 100, 2),
                        round(backtest_results['trade_stats']['profit_factor'], 2)
                    ]
                })
                st.dataframe(trade_stats)

                # Final positions
                st.subheader("Final Positions")
                positions_df = pd.DataFrame({
                    'Ticker': list(backtest_results['final_positions'].keys()),
                    'Shares': [pos['shares'] for pos in backtest_results['final_positions'].values()],
                    'Market Value ($)': [pos['market_value'] for pos in backtest_results['final_positions'].values()],
                    'Weight (%)': [round(pos['weight'] * 100, 2) for pos in backtest_results['final_positions'].values()]
                })
                st.dataframe(positions_df)
            
    with tabs[4]:
        st.header("Execution")
        broker_type = st.selectbox("Broker", ["Paper Trading", "Zerodha"])
        
        if st.button("Execute Strategy"):
            from Algo_trading import PaperBrokerAdapter, PaperTrader
            with st.spinner("Executing strategy..."):
                initial_capital = st.session_state['initial_capital']
                if broker_type == "Paper Trading":
                    paper = PaperTrader(cash=initial_capital)
                    broker = PaperBrokerAdapter(paper)      
                else:  # Zerodha
                    st.warning("Zerodha integration requires API credentials.")
                    paper = PaperTrader(cash=initial_capital)
                    broker = PaperBrokerAdapter(paper)  # Fallback to paper
                
                # Execute orders based on optimized weights
                for ticker, weight in zip(st.session_state['selected_tickers'], st.session_state['optimized_weights']):
                    if weight > 0:
                        price = st.session_state['data'][ticker]['Close'].iloc[-1]
                        if isinstance(price, pd.Series):
                            price = price.iloc[0]
                        shares = int((weight * initial_capital) / price)
                        
                        if shares > 0:
                            order_result = broker.place_order(
                                ticker=ticker,
                                order_type="market",
                                side="buy",
                                qty=shares,
                                price=price
                            )
                            st.success(f"Placed order for {shares} shares of {ticker}")
                st.success("Strategy execution completed!")
    # Results (Step 5)
    with tabs[5]:
        st.header("Results")
        
        if 'backtest_results' in st.session_state:
            # Calculate performance metrics
            equity_curve = st.session_state['backtest_results']['equity_curve']
            from backtest import compute_performance_metrics
            metrics = compute_performance_metrics(equity_curve)
            
            # Display performance metrics
            st.subheader("Performance Metrics")
            metrics_df = pd.DataFrame({
                'Metric': ['CAGR (%)', 'Annual Volatility (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Calmar Ratio'],
                'Value': [
                    round(metrics['cagr'] * 100, 2),
                    round(metrics['annual_volatility'] * 100, 2),
                    round(metrics['sharpe_ratio'], 2),
                    round(metrics['max_drawdown'] * 100, 2),
                    round(metrics['calmar_ratio'], 2)
                ]
            })
            st.dataframe(metrics_df)
            
            # Plot equity curve and drawdown
            st.subheader("Equity Curve and Drawdown")
            
            # Create two subplots
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.1, 
                               subplot_titles=('Equity Curve', 'Drawdown'))
            
            # Add equity curve
            fig.add_trace(
                go.Scatter(x=equity_curve.index, y=equity_curve, mode='lines', name='Equity'),
                row=1, col=1
            )
            
            # Calculate and add drawdown
            drawdown = (equity_curve / equity_curve.cummax() - 1) * 100
            fig.add_trace(
                go.Scatter(x=drawdown.index, y=drawdown, mode='lines', 
                          name='Drawdown', line=dict(color='red')),
                row=2, col=1
            )
            
            fig.update_layout(height=600, showlegend=True)
            fig.update_yaxes(title_text='Portfolio Value ($)', row=1, col=1)
            fig.update_yaxes(title_text='Drawdown (%)', row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Provide download link for results
            st.subheader("Download Results")
            
            # Prepare results for download
            results_df = pd.DataFrame({
                'Date': equity_curve.index,
                'Equity': equity_curve.values,
                'Drawdown (%)': drawdown.values
            })
            
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="backtest_results.csv",
                mime="text/csv"
            )
            
            # Add a conclusion
            st.subheader("Conclusion")
            st.write("""
            This backtest represents a simulation of the optimized portfolio strategy. 
            The results shown are based on historical data and do not guarantee future performance.
            
            Key takeaways:
            - The strategy achieved a CAGR of {:.2f}%
            - Maximum drawdown was {:.2f}%
            - The Sharpe ratio of {:.2f} indicates the risk-adjusted return
            
            Consider running additional tests with different parameters or time periods to validate the robustness of the strategy.
            """.format(
                metrics['cagr'] * 100,
                metrics['max_drawdown'] * 100,
                metrics['sharpe_ratio']
            ))
            
        else:
            st.info("Please run a backtest first in the Backtest & Execution tab.")

else:
    # Show warning if no tickers selected
    st.warning("Please select at least one ticker in the sidebar to begin.")