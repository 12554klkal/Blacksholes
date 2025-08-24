import streamlit as st
import numpy as np
import pandas as pd
from math import log, sqrt, exp
from scipy.stats import norm
from scipy.optimize import brentq
import plotly.graph_objects as go
import yfinance as yf

# -------------------------
# Black-Scholes & Implied Volatility Functions
# -------------------------
def bs_price(S, K, T, r, sigma, option_type="call"):
    """Black-Scholes formula to calculate option price."""
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    else:
        return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def find_implied_volatility(market_price, S, K, T, r, option_type):
    """Finds implied volatility using Brent's method."""
    if T <= 0 or market_price <= 0:
        return np.nan
    try:
        func = lambda sigma: bs_price(S, K, T, r, sigma, option_type) - market_price
        implied_vol = brentq(func, 0.0001, 10)
        return implied_vol
    except:
        return np.nan

def calculate_historical_volatility(stock_symbol):
    """Calculates annualized historical volatility from yfinance data."""
    try:
        data = yf.download(stock_symbol, period="1y", interval="1d", progress=False)
        returns = np.log(data['Adj Close'] / data['Adj Close'].shift(1)).dropna()
        volatility = returns.std() * np.sqrt(252)
        return volatility
    except:
        return None

# -------------------------
# Streamlit Layout
# -------------------------
st.set_page_config(layout="wide")
st.title("Black-Scholes Pricing Model")

# Sidebar for inputs
st.sidebar.markdown("### Connect with me")
st.sidebar.markdown(
    "[📎 LinkedIn](https://www.linkedin.com/in/jonas-f-628179296/)",
    unsafe_allow_html=True
)

st.sidebar.header("Asset Parameters")

input_method = st.sidebar.radio("Select Input Method", ("Search for a Stock", "Enter Manually"))

S, sigma = 100.0, 0.2  # Set default values

if input_method == "Search for a Stock":
    stock_symbol = st.sidebar.text_input("Enter a stock symbol (e.g., AAPL, GOOGL)", 'AAPL').upper()
    
    if stock_symbol:
        ticker = yf.Ticker(stock_symbol)
        try:
            S = ticker.info['regularMarketPrice']
            st.sidebar.markdown(f"**Current Price:** ${S:.2f}")

            sigma = calculate_historical_volatility(stock_symbol)
            if sigma is not None:
                st.sidebar.markdown(f"**Historical Volatility:** {sigma:.2%}")
            else:
                st.sidebar.warning("Could not calculate historical volatility. Please enter manually.")
                sigma = st.sidebar.number_input("Manual Volatility (σ)", value=0.2, step=0.01, min_value=0.01)

            # Add the link to the volatility search page
            volatility_link = f"https://www.alphaquery.com/stock/{stock_symbol}/volatility-option-statistics/30-day/historical-volatility"
            st.sidebar.markdown(f"[📊 Search for Historic Volatility]({volatility_link})")

        except (KeyError, IndexError):
            st.sidebar.error("Invalid stock symbol or could not fetch data.")
            S = st.sidebar.number_input("Current Asset Price", value=100.0, step=1.0)
            sigma = st.sidebar.number_input("Volatility (σ)", value=0.2, step=0.01, min_value=0.01)

else:  # "Enter Manually"
    S = st.sidebar.number_input("Current Asset Price", value=100.0, step=1.0)
    sigma = st.sidebar.number_input("Volatility (σ)", value=0.2, step=0.01, min_value=0.01)

st.sidebar.header("Option Parameters")
K = st.sidebar.number_input("Strike Price", value=100.0, step=1.0)
T = st.sidebar.number_input("Time to Maturity (Years)", value=1.0, step=0.1, min_value=0.01)
r = st.sidebar.number_input("Risk-Free Interest Rate", value=0.05, step=0.01)

# -------------------------
# Display single option prices
# -------------------------
st.subheader("Current Option Prices")
col1, col2 = st.columns(2)
with col1:
    call_value = bs_price(S, K, T, r, sigma, 'call')
    st.metric("CALL Value", f"${call_value:.2f}")
with col2:
    put_value = bs_price(S, K, T, r, sigma, 'put')
    st.metric("PUT Value", f"${put_value:.2f}")

# -------------------------
# Heatmap Parameters and calculations
# -------------------------
st.subheader("Options Price - Interactive Heatmaps")
st.sidebar.header("Heatmap Parameters")
default_s_min = max(1.0, S - 20)
default_s_max = S + 20
S_min = st.sidebar.number_input("Min Spot Price", value=default_s_min, step=1.0)
S_max = st.sidebar.number_input("Max Spot Price", value=default_s_max, step=1.0)
st.sidebar.markdown("---")
st.sidebar.subheader("Volatility Range for Heatmap")
sigma_min = st.sidebar.slider("Min Volatility", 0.01, 1.0, 0.1, step=0.01)
sigma_max = st.sidebar.slider("Max Volatility", 0.01, 1.0, 0.3, step=0.01)
spot_prices = np.linspace(S_min, S_max, 10)
vols = np.linspace(sigma_min, sigma_max, 9)
call_prices = np.zeros((len(vols), len(spot_prices)))
put_prices = np.zeros((len(vols), len(spot_prices)))
for i, v in enumerate(vols):
    for j, s in enumerate(spot_prices):
        if s > 0:
            call_prices[i, j] = bs_price(s, K, T, r, v, "call")
            put_prices[i, j] = bs_price(s, K, T, r, v, "put")
call_min_price, call_max_price = call_prices.min(), call_prices.max()
put_min_price, put_max_price = put_prices.min(), put_prices.max()

def create_heatmap(prices, x_labels, y_labels, title, min_price, max_price):
    custom_colorscale = [
        [0.0, 'rgb(180, 219, 203)'],
        [0.2, 'rgb(102, 184, 159)'],
        [0.4, 'rgb(240, 240, 240)'],
        [0.6, 'rgb(255, 150, 150)'],
        [0.8, 'rgb(255, 100, 100)'],
        [1.0, 'rgb(255, 50, 50)']
    ]
    fig = go.Figure(data=go.Heatmap(
        z=prices,
        x=x_labels,
        y=y_labels,
        colorscale=custom_colorscale,
        colorbar=dict(title="Price", thickness=20),
        zmin=min_price,
        zmax=max_price
    ))
    annotations = []
    mid_point = np.median(prices)
    for i, vol in enumerate(y_labels):
        for j, spot in enumerate(x_labels):
            annotations.append(
                dict(
                    x=spot,
                    y=vol,
                    text=f"{prices[i, j]:.2f}",
                    showarrow=False,
                    font=dict(color="white" if prices[i,j] > mid_point else "black")
                )
            )
    fig.update_layout(
        title=title,
        xaxis=dict(title="Spot Price", tickmode='array', tickvals=x_labels, ticktext=[f'{p:.2f}' for p in x_labels]),
        yaxis=dict(title="Volatility", tickmode='array', tickvals=y_labels, ticktext=[f'{v:.2f}' for v in y_labels]),
        annotations=annotations,
    )
    return fig

# -------------------------
# Create and display heatmaps
# -------------------------
col3, col4 = st.columns(2)
with col3:
    fig_call = create_heatmap(call_prices, spot_prices, vols, "CALL", call_min_price, call_max_price)
    st.plotly_chart(fig_call, use_container_width=True)
with col4:
    fig_put = create_heatmap(put_prices, spot_prices, vols, "PUT", put_min_price, put_max_price)
    st.plotly_chart(fig_put, use_container_width=True)

# -------------------------
# New Feature: Options Screener
# -------------------------
st.subheader("Highest Potential Profit Screener")
st.sidebar.subheader("Screener Parameters")
if input_method == "Search for a Stock" and stock_symbol:
    if S is not None:
        target_price = st.sidebar.number_input(
            "Your Target Stock Price", 
            value=float(S), 
            step=1.0, 
            help="Enter the price you believe the stock will reach by expiry."
        )

if input_method == "Search for a Stock" and stock_symbol and S is not None:
    try:
        expiries = yf.Ticker(stock_symbol).options
        if not expiries:
            st.warning("No option chain data available for this stock.")
        else:
            exp_date = expiries[0]
            st.info(f"Analyzing options for expiry: **{exp_date}**")

            chain = yf.Ticker(stock_symbol).option_chain(exp_date)
            calls = chain.calls
            puts = chain.puts
            
            profitable_options = []
            
            # Time to expiry in years
            days_to_expiry = (pd.to_datetime(exp_date) - pd.Timestamp.now()).days
            if days_to_expiry <= 0:
                st.warning("Expiration date is in the past or today. Cannot calculate profit.")
            else:
                T_days = days_to_expiry / 365.0

                # Analyze call options
                for _, row in calls.iterrows():
                    market_price = row.get('lastPrice', row.get('bid'))
                    strike = row['strike']

                    if not pd.isna(market_price) and market_price > 0:
                        # Calculate theoretical value at target price
                        theoretical_value_at_target = bs_price(target_price, strike, T_days, r, sigma, 'call')
                        
                        potential_profit = theoretical_value_at_target - market_price
                        
                        if potential_profit > 0:
                            # Calculate ROI to find the most leveraged options
                            roi = potential_profit / market_price
                            profitable_options.append({
                                'Type': 'Call',
                                'Strike': strike,
                                'Market Price': f"${market_price:.2f}",
                                'Potential Profit': f"${potential_profit:.2f}",
                                'ROI': f"{roi:.2%}",
                                'Link': f"https://finance.yahoo.com/quote/{row['contractSymbol']}",
                                'sort_roi': roi
                            })

                # Analyze put options
                for _, row in puts.iterrows():
                    market_price = row.get('lastPrice', row.get('bid'))
                    strike = row['strike']

                    if not pd.isna(market_price) and market_price > 0:
                        # Calculate theoretical value at target price
                        theoretical_value_at_target = bs_price(target_price, strike, T_days, r, sigma, 'put')

                        potential_profit = theoretical_value_at_target - market_price
                        
                        if potential_profit > 0:
                            # Calculate ROI to find the most leveraged options
                            roi = potential_profit / market_price
                            profitable_options.append({
                                'Type': 'Put',
                                'Strike': strike,
                                'Market Price': f"${market_price:.2f}",
                                'Potential Profit': f"${potential_profit:.2f}",
                                'ROI': f"{roi:.2%}",
                                'Link': f"https://finance.yahoo.com/quote/{row['contractSymbol']}",
                                'sort_roi': roi
                            })

                if profitable_options:
                    st.write(f"Top 10 options with the highest ROI for a target price of **${target_price:.2f}**:")
                    df_profitable = pd.DataFrame(profitable_options)
                    df_profitable = df_profitable.sort_values(by='sort_roi', ascending=False).head(10)
                    df_profitable['Link'] = df_profitable['Link'].apply(lambda x: f'<a href="{x}" target="_blank">View on Yahoo Finance</a>')
                    
                    # Drop the temporary column used for sorting
                    df_profitable = df_profitable.drop(columns=['sort_roi'])
                    
                    st.markdown(df_profitable.to_html(escape=False), unsafe_allow_html=True)
                else:
                    st.info("No options with a positive potential profit found based on your target price.")
    
    except Exception as e:
        st.error(f"Error fetching option chain data: {e}")
