import math
from typing import Literal, Tuple

import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ---------- Black–Scholes core ----------

def _d1_d2(S: float, K: float, r: float, q: float, sigma: float, T: float) -> Tuple[float, float]:
    if sigma <= 0 or T <= 0:
        # Avoid division by zero; return extreme values that make CDFs snap to 0/1 appropriately
        sign = 1e9 if (S > K) else -1e9
        return sign, sign - sigma * math.sqrt(max(T, 1e-16))
    vol_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2

# Standard normal pdf and cdf
def _phi(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def _N(x: float) -> float:
    # stable CDF via erf
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

def bs_price(option: Literal["Call", "Put"], S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    d1, d2 = _d1_d2(S, K, r, q, sigma, T)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    if option == "Call":
        return disc_q * S * _N(d1) - disc_r * K * _N(d2)
    else:
        return disc_r * K * _N(-d2) - disc_q * S * _N(-d1)

def greeks(option: Literal["Call", "Put"], S: float, K: float, r: float, q: float, sigma: float, T: float):
    d1, d2 = _d1_d2(S, K, r, q, sigma, T)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    delta = disc_q * _N(d1) if option == "Call" else disc_q * (_N(d1) - 1)
    gamma = disc_q * _phi(d1) / (S * sigma * math.sqrt(T)) if sigma > 0 and T > 0 else 0.0
    vega = S * disc_q * _phi(d1) * math.sqrt(T) / 100.0  # per 1% vol
    # Theta per day (calendar day)
    if option == "Call":
        theta = (-S * disc_q * _phi(d1) * sigma / (2 * math.sqrt(T))
                 - r * K * disc_r * _N(d2)
                 + q * S * disc_q * _N(d1)) / 365.0
    else:
        theta = (-S * disc_q * _phi(d1) * sigma / (2 * math.sqrt(T))
                 + r * K * disc_r * _N(-d2)
                 - q * S * disc_q * _N(-d1)) / 365.0
    rho = (K * T * disc_r * _N(d2) / 100.0) if option == "Call" else (-K * T * disc_r * _N(-d2) / 100.0)  # per 1% rate
    return {
        "Delta": delta,
        "Gamma": gamma,
        "Vega (per 1% σ)": vega,
        "Theta/day": theta,
        "Rho (per 1% r)": rho
    }

def implied_vol(option: Literal["Call", "Put"], S: float, K: float, r: float, q: float, T: float, target_price: float,
                tol: float = 1e-6, max_iter: int = 100) -> float:
    # Robust bisection on [low, high]
    # Start with wide bounds that cover practical vols
    low, high = 1e-6, 5.0
    # Ensure target is bracketed
    for _ in range(50):
        pl = bs_price(option, S, K, r, q, low, T)
        ph = bs_price(option, S, K, r, q, high, T)
        if (pl - target_price) * (ph - target_price) <= 0:
            break
        high *= 2
        if high > 100:
            return float("nan")
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        pm = bs_price(option, S, K, r, q, mid, T)
        diff = pm - target_price
        if abs(diff) < tol:
            return mid
        # Decide which side to keep
        if (bs_price(option, S, K, r, q, low, T) - target_price) * diff < 0:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)

# ---------- UI ----------

st.set_page_config(page_title="Black–Scholes Option Pricing", page_icon="📈", layout="wide")

st.title("Black–Scholes Option Pricing Model")
st.caption("European options with continuous dividend yield. Pricing, Greeks, heatmaps, and implied volatility.")

with st.sidebar:
    st.header("Inputs")
    col1, col2 = st.columns(2)
    with col1:
        option_type = st.selectbox("Option type", ["Call", "Put"])
        S = st.number_input("Spot price S", min_value=0.01, value=100.0, step=1.0, format="%.2f")
        K = st.number_input("Strike K", min_value=0.01, value=100.0, step=1.0, format="%.2f")
        T_years = st.number_input("Time to expiry (years)", min_value=0.0001, value=0.5, step=0.05, format="%.4f")
    with col2:
        r_pct = st.number_input("Risk-free rate r (%)", value=3.0, step=0.25, format="%.2f")
        q_pct = st.number_input("Dividend yield q (%)", value=0.0, step=0.25, format="%.2f")
        sigma_pct = st.number_input("Volatility σ (%)", min_value=0.01, value=20.0, step=0.5, format="%.2f")
        show_iv = st.checkbox("Compute implied volatility from a target premium", value=False)
    r = r_pct / 100.0
    q = q_pct / 100.0
    sigma = sigma_pct / 100.0

    if show_iv:
        target = st.number_input("Target option premium", min_value=0.0, value=float(max(1.0, abs(S-K)*0.25)), step=0.25)
        st.caption("IV is solved via robust bisection on [1e-6, 5.0].")

# Top metrics
price = bs_price(option_type, S, K, r, q, sigma, T_years)
g = greeks(option_type, S, K, r, q, sigma, T_years)

m1, m2, m3 = st.columns(3)
m1.metric("Option price", f"{price:,.4f}")
m2.metric("Delta", f"{g['Delta']:.4f}")
m3.metric("Gamma", f"{g['Gamma']:.6f}")

m4, m5, m6 = st.columns(3)
m4.metric("Vega (per 1% σ)", f"{g['Vega (per 1% σ)']:.4f}")
m5.metric("Theta / day", f"{g['Theta/day']:.4f}")
m6.metric("Rho (per 1% r)", f"{g['Rho (per 1% r)']:.4f}")

if show_iv:
    iv = implied_vol(option_type, S, K, r, q, T_years, target)
    st.info(f"Implied volatility for premium {target:.4f}: **{(iv*100):.3f}%**")

st.markdown("---")

# ---------- Charts row 1: price vs spot; price vs vol ----------
c1, c2 = st.columns(2)

# Price vs Spot
with c1:
    S_grid = np.linspace(max(0.01, 0.1 * S), 2.0 * S, 200)
    prices_vs_S = [bs_price(option_type, float(sv), K, r, q, sigma, T_years) for sv in S_grid]
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=S_grid, y=prices_vs_S, mode="lines", name=f"{option_type} price"))
    fig1.update_layout(title="Price vs Spot", xaxis_title="Spot S", yaxis_title="Option price", height=420)
    st.plotly_chart(fig1, use_container_width=True)

# Price vs Vol
with c2:
    vol_grid = np.linspace(0.01, 1.0, 200)  # 1% to 100% annualized
    prices_vs_vol = [bs_price(option_type, S, K, r, q, float(v), T_years) for v in vol_grid]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=vol_grid * 100.0, y=prices_vs_vol, mode="lines", name=f"{option_type} price"))
    fig2.update_layout(title="Price vs Volatility", xaxis_title="Volatility σ (%)", yaxis_title="Option price", height=420)
    st.plotly_chart(fig2, use_container_width=True)

# ---------- Charts row 2: heatmaps ----------
st.subheader("Heatmap: price across Spot and Volatility")
with st.expander("Heatmap settings", expanded=True):
    colh1, colh2, colh3 = st.columns(3)
    with colh1:
        s_min = st.number_input("S min", min_value=0.01, value=float(max(0.1 * S, 1e-2)))
        s_max = st.number_input("S max", min_value=s_min + 1e-3, value=float(2.0 * S))
    with colh2:
        v_min_pct = st.number_input("σ min (%)", min_value=0.01, value=5.0, step=0.5)
        v_max_pct = st.number_input("σ max (%)", min_value=v_min_pct + 0.01, value=80.0, step=0.5)
    with colh3:
        n_spot = st.slider("Spot steps", min_value=20, max_value=200, value=80, step=10)
        n_vol = st.slider("Vol steps", min_value=20, max_value=200, value=60, step=10)

S_vals = np.linspace(s_min, s_max, n_spot)
V_vals = np.linspace(v_min_pct / 100.0, v_max_pct / 100.0, n_vol)
Z = np.zeros((n_vol, n_spot))
for i, vv in enumerate(V_vals):
    for j, ss in enumerate(S_vals):
        Z[i, j] = bs_price(option_type, float(ss), K, r, q, float(vv), T_years)

fig_hm = go.Figure(
    data=go.Heatmap(
        z=Z,
        x=S_vals,
        y=V_vals * 100.0,
        colorbar=dict(title="Price"),
        zsmooth="best",
    )
)
fig_hm.update_layout(
    title=f"{option_type} price heatmap",
    xaxis_title="Spot S",
    yaxis_title="Volatility σ (%)",
    height=560,
)
st.plotly_chart(fig_hm, use_container_width=True)

# ---------- Footer ----------
st.caption(
    "Notes: European BSM with continuous dividend yield. Theta shown per calendar day. "
    "Vega and Rho reported per 1 percentage-point change in σ and r respectively."
)
