import math
import numpy as np
import streamlit as st
import plotly.express as px

# ---------- Black–Scholes core ----------
def _d1_d2(S, K, r, sigma, T):
    vol_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2

def _N(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def bs_price(option, S, K, r, sigma, T):
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    if option == "Call":
        return S * _N(d1) - K * math.exp(-r * T) * _N(d2)
    else:
        return K * math.exp(-r * T) * _N(-d2) - S * _N(-d1)

# ---------- UI ----------
st.set_page_config(page_title="Black–Scholes Pricing Model", layout="wide")

st.title("Black–Scholes Pricing Model")

# Inputs on the RIGHT
with st.sidebar:
    st.header("Parameters")
    S = st.number_input("Current Asset Price", value=100.0, step=1.0)
    K = st.number_input("Strike Price", value=100.0, step=1.0)
    T = st.number_input("Time to Maturity (Years)", value=1.0, step=0.1)
    sigma = st.number_input("Volatility (σ)", value=0.20, step=0.01)
    r = st.number_input("Risk-Free Interest Rate", value=0.05, step=0.01)

    st.header("Heatmap Parameters")
    S_min = st.number_input("Min Spot Price", value=80.0, step=1.0)
    S_max = st.number_input("Max Spot Price", value=120.0, step=1.0)
    v_min = st.number_input("Min Volatility for Heatmap", value=0.10, step=0.01)
    v_max = st.number_input("Max Volatility for Heatmap", value=0.30, step=0.01)

# Display single values (call & put at the entered params)
call_value = bs_price("Call", S, K, r, sigma, T)
put_value = bs_price("Put", S, K, r, sigma, T)

col1, col2 = st.columns(2)
col1.success(f"CALL Value: ${call_value:.2f}")
col2.error(f"PUT Value: ${put_value:.2f}")

st.markdown("## Options Price – Interactive Heatmap")

# Heatmap grid
spot_range = np.linspace(S_min, S_max, 20)
vol_range = np.linspace(v_min, v_max, 20)

call_prices = np.zeros((len(vol_range), len(spot_range)))
put_prices = np.zeros_like(call_prices)

for i, v in enumerate(vol_range):
    for j, s in enumerate(spot_range):
        call_prices[i, j] = bs_price("Call", s, K, r, v, T)
        put_prices[i, j] = bs_price("Put", s, K, r, v, T)

# Heatmap plots (green→red)
fig_call = px.imshow(
    call_prices,
    x=np.round(spot_range, 2),
    y=np.round(vol_range, 2),
    color_continuous_scale=["green", "red"],
    labels=dict(x="Spot Price", y="Volatility", color="CALL"),
    aspect="auto",
)
fig_put = px.imshow(
    put_prices,
    x=np.round(spot_range, 2),
    y=np.round(vol_range, 2),
    color_continuous_scale=["green", "red"],
    labels=dict(x="Spot Price", y="Volatility", color="PUT"),
    aspect="auto",
)

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(fig_call, use_container_width=True)
with c2:
    st.plotly_chart(fig_put, use_container_width=True)
