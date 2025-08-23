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

st.sidebar.header("Heatmap Parameters")

# Fix: Ensure S_min is always greater than 0
S_min_default = S - 20
S_min = st.sidebar.number_input("Min Spot Price", value=max(1.0, S_min_default), step=1.0, min_value=1.0)
S_max = st.sidebar.number_input("Max Spot Price", value=S + 20, step=1.0)

# Range slider for volatility
vol_range = st.sidebar.slider("Volatility Range for Heatmap", 0.01, 1.0, (0.1, 0.3), step=0.01)
sigma_min, sigma_max = vol_range

# -------------------------
# Display single option prices
# -------------------------
col1, col2 = st.columns(2)
with col1:
    st.metric("CALL Value", f"${bs_price(S, K, T, r, sigma, 'call'):.2f}")
with col2:
    st.metric("PUT Value", f"${bs_price(S, K, T, r, sigma, 'put'):.2f}")

# -------------------------
# Heatmaps calculations
# -------------------------
spot_prices = np.linspace(S_min, S_max, 10)
vols = np.linspace(sigma_min, sigma_max, 9)

call_prices = np.zeros((len(vols), len(spot_prices)))
put_prices = np.zeros((len(vols), len(spot_prices)))

for i, v in enumerate(vols):
    for j, s in enumerate(spot_prices):
        call_prices[i, j] = bs_price(s, K, T, r, v, "call")
        put_prices[i, j] = bs_price(s, K, T, r, v, "put")

# -------------------------
# Heatmap function
# -------------------------
def create_heatmap(prices, x_labels, y_labels, title):
    
    # Custom color scale to match the provided image
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
        zmin=np.min(prices),
        zmax=np.max(prices) 
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
st.subheader("Options Price - Interactive Heatmaps")
col3, col4 = st.columns(2)
with col3:
    fig_call = create_heatmap(call_prices, spot_prices, vols, "CALL")
    st.plotly_chart(fig_call, use_container_width=True)
with col4:
    fig_put = create_heatmap(put_prices, spot_prices, vols, "PUT")
    st.plotly_chart(fig_put, use_container_width=True)
