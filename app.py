import streamlit as st
import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm
import plotly.graph_objects as go

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
    "[📎 LinkedIn](https://www.linkedin.com/in/jonas-f-628179296/)",
    unsafe_allow_html=True
)

st.sidebar.header("Input Parameters")

S = st.sidebar.number_input("Current Asset Price", value=100.0, step=1.0)
K = st.sidebar.number_input("Strike Price", value=100.0, step=1.0)
T = st.sidebar.number_input("Time to Maturity (Years)", value=1.0, step=0.1, min_value=0.01)
sigma = st.sidebar.number_input("Volatility (σ)", value=0.2, step=0.01, min_value=0.01)
r = st.sidebar.number_input("Risk-Free Interest Rate", value=0.05, step=0.01)

# -------------------------
# Display single option prices
# -------------------------
col1, col2 = st.columns(2)
with col1:
    st.metric("CALL Value", f"${bs_price(S, K, T, r, sigma, 'call'):.2f}")
with col2:
    st.metric("PUT Value", f"${bs_price(S, K, T, r, sigma, 'put'):.2f}")

# -------------------------
# Heatmap Parameters and calculations
# -------------------------
st.subheader("Options Price - Interactive Heatmaps")

# Dynamically calculate S_min and S_max based on the current asset price (S)
# This centers the heatmap on the current price.
s_range_diff = 40.0 # A fixed range for the heatmap, you can adjust this
S_min = max(1.0, S - s_range_diff / 2) # Ensure S_min is never negative or zero
S_max = S + s_range_diff / 2

# Dynamically calculate the volatility range based on the current asset price
# Higher-priced stocks might have a different volatility range.
vol_range_max = 0.5 if S < 200 else 0.3
sigma_min_val = 0.1
sigma_max_val = st.slider("Max Volatility for Heatmap", 0.01, vol_range_max, 0.3, step=0.01)

# Generate the data points for the heatmap
spot_prices = np.linspace(S_min, S_max, 10)
vols = np.linspace(sigma_min_val, sigma_max_val, 9)

# Calculate prices for the heatmap grid
call_prices = np.zeros((len(vols), len(spot_prices)))
put_prices = np.zeros((len(vols), len(spot_prices)))

for i, v in enumerate(vols):
    for j, s in enumerate(spot_prices):
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



