import streamlit as st
import numpy as np
import pandas as pd
from math import log, sqrt, exp
from scipy.stats import norm
import plotly.graph_objects as go
import yfinance as yf

# -------------------------
# Black-Scholes Functions
# -------------------------
def bs_price(S, K, T, r, sigma, option="call"):
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if option == "call":
        return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    else:
        return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Function to calculate historical volatility
def calculate_volatility(stock_symbol):
    try:
        data = yf.download(stock_symbol, period="1y", interval="1d", progress=False)
        returns = np.log(data['Adj Close'] / data['Adj Close'].shift(1)).dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualize volatility
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

            sigma = calculate_volatility(stock_symbol)
            if sigma is not None:
                st.sidebar.markdown(f"**Historical Volatility:** {sigma:.2%}")
            else:
                st.sidebar.warning("Could not calculate volatility. Please enter manually.")
                sigma = st.sidebar.number_input("Manual Volatility (σ)", value=0.2, step=0.01, min_value=0.01)

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
# Dynamically set default values but allow manual override
default_s_min = max(1.0, S - 20)
default_s_max = S + 20
S_min = st.sidebar.number_input("Min Spot Price", value=default_s_min, step=1.0)
S_max = st.sidebar.number_input("Max Spot Price", value=default_s_max, step=1.0)

# Add the volatility sliders back to the sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Volatility Range for Heatmap")
sigma_min = st.sidebar.slider("Min Volatility", 0.01, 1.0, 0.1, step=0.01)
sigma_max = st.sidebar.slider("Max Volatility", 0.01, 1.0, 0.3, step=0.01)

# Generate the data points for the heatmap
spot_prices = np.linspace(S_min, S_max, 10)
vols = np.linspace(sigma_min, sigma_max, 9)

# Calculate prices for the heatmap grid
call_prices = np.zeros((len(vols), len(spot_prices)))
put_prices = np.zeros((len(vols), len(spot_prices)))

for i, v in enumerate(vols):
    for j, s in enumerate(spot_prices):
        # Add a check to prevent log(0) or log(negative)
        if s > 0:
            call_prices[i, j] = bs_price(s, K, T, r, v, "call")
            put_prices[i, j] = bs_price(s, K, T, r, v, "put")

# Dynamic min and max for the color scale
call_min_price, call_max_price = call_prices.min(), call_prices.max()
put_min_price, put_max_price = put_prices.min(), put_prices.max()

# -------------------------
# Heatmap function
# -------------------------
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

