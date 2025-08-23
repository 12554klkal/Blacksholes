import streamlit as st
import numpy as np
import plotly.express as px
from math import log, sqrt, exp
from scipy.stats import norm

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

# -------------------------
# Streamlit Layout
# -------------------------
st.set_page_config(layout="wide")
st.title("Black-Scholes Pricing Model")

# Sidebar for inputs
st.sidebar.markdown("### Connect with me")
st.sidebar.markdown(
    "[📎 LinkedIn](https://www.linkedin.com/in/jonas-f-628179296?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app)",
    unsafe_allow_html=True
)

st.sidebar.header("Input Parameters")

S = st.sidebar.number_input("Current Asset Price", value=100.0, step=1.0)
K = st.sidebar.number_input("Strike Price", value=100.0, step=1.0)
T = st.sidebar.number_input("Time to Maturity (Years)", value=1.0, step=0.1, min_value=0.01)
sigma = st.sidebar.number_input("Volatility (σ)", value=0.2, step=0.01, min_value=0.01)
r = st.sidebar.number_input("Risk-Free Interest Rate", value=0.05, step=0.01)

st.sidebar.header("Heatmap Parameters")
S_min = st.sidebar.number_input("Min Spot Price", value=80.0, step=1.0)
S_max = st.sidebar.number_input("Max Spot Price", value=120.0, step=1.0)

# Range slider for volatility
vol_range = st.sidebar.slider("Volatility Range for Heatmap", 0.01, 1.0, (0.1, 0.3), step=0.01)
sigma_min, sigma_max = vol_range

# -------------------------
# Calculate and Display
# -------------------------
col1, col2 = st.columns(2)

with col1:
    st.metric("CALL Value", f"${bs_price(S, K, T, r, sigma, 'call'):.2f}")

with col2:
    st.metric("PUT Value", f"${bs_price(S, K, T, r, sigma, 'put'):.2f}")

# -------------------------
# Heatmaps
# -------------------------
spot_prices = np.linspace(S_min, S_max, 15)
vols = np.linspace(sigma_min, sigma_max, 15)

call_prices = np.zeros((len(vols), len(spot_prices)))
put_prices = np.zeros((len(vols), len(spot_prices)))

for i, v in enumerate(vols):
    for j, s in enumerate(spot_prices):
        call_prices[i, j] = bs_price(s, K, T, r, v, "call")
        put_prices[i, j] = bs_price(s, K, T, r, v, "put")

# Plotly heatmaps with visible gridlines + text
fig_call = px.imshow(
    call_prices,
    x=np.round(spot_prices, 2),
    y=np.round(vols, 2),
    color_continuous_scale="RdYlGn_r",
    text_auto=".2f",
    aspect="auto"
)
fig_call.update_layout(title="Call Price Heatmap", width=700, height=600)
fig_call.update_traces(hovertemplate="Spot=%{x}, Vol=%{y}<br>Call=%{z:.2f}")

fig_put = px.imshow(
    put_prices,
    x=np.round(spot_prices, 2),
    y=np.round(vols, 2),
    color_continuous_scale="RdYlGn_r",
    text_auto=".2f",
    aspect="auto"
)
fig_put.update_layout(title="Put Price Heatmap", width=700, height=600)
fig_put.update_traces(hovertemplate="Spot=%{x}, Vol=%{y}<br>Put=%{z:.2f}")

st.subheader("Options Price - Interactive Heatmaps")
col3, col4 = st.columns(2)
col3.plotly_chart(fig_call, use_container_width=True)
col4.plotly_chart(fig_put, use_container_width=True)
