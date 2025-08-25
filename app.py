import streamlit as st
import numpy as np
import pandas as pd
from math import log, sqrt, exp, pi
from scipy.stats import norm
from scipy.optimize import brentq
import plotly.graph_objects as go
import yfinance as yf

# -------------------------
# Black-Scholes & Implied Volatility Functions
# -------------------------
def d1(S, K, T, r, sigma):
    """Calculates d1 for the Black-Scholes formula."""
    # Ensure sigma and T are not zero to prevent division by zero
    if sigma == 0 or T == 0:
        return np.inf if S > K else -np.inf # Handle deterministic case or immediate expiry
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def d2(S, K, T, r, sigma):
    """Calculates d2 for the Black-Scholes formula."""
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def bs_price(S, K, T, r, sigma, option_type="call"):
    """Black-Scholes formula to calculate option price."""
    # Handle edge case where time to maturity is zero or negative
    if T <= 0:
        if option_type == "call":
            return max(0.0, S - K)
        else:
            return max(0.0, K - S)

    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)

    if option_type == "call":
        return S * norm.cdf(d1_val) - K * exp(-r * T) * norm.cdf(d2_val)
    else:
        return K * exp(-r * T) * norm.cdf(-d2_val) - S * norm.cdf(-d1_val)

def find_implied_volatility(market_price, S, K, T, r, option_type):
    """Finds implied volatility using Brent's method."""
    if T <= 0 or market_price <= 0:
        return np.nan
    
    # Check for arbitrage bounds to make brentq more robust
    if option_type == "call":
        lower_bound_price = max(0, S - K * exp(-r * T))
        if market_price < lower_bound_price:
            return np.nan # Arbitrage opportunity, IV cannot be found
    else: # put
        lower_bound_price = max(0, K * exp(-r * T) - S)
        if market_price < lower_bound_price:
            return np.nan # Arbitrage opportunity, IV cannot be found

    try:
        # Define the function whose root we want to find
        func = lambda sigma: bs_price(S, K, T, r, sigma, option_type) - market_price
        
        # Use brentq to find the root within a reasonable range for volatility (0.0001 to 10)
        # Ensure that the function changes sign over the interval [0.0001, 10]
        # This check is crucial for brentq to work
        if func(0.0001) * func(10) < 0:
            implied_vol = brentq(func, 0.0001, 10)
            return implied_vol
        else:
            return np.nan # Function does not cross zero in the interval
    except:
        return np.nan

# -------------------------
# Option Greeks Calculations
# -------------------------
def delta(S, K, T, r, sigma, option_type="call"):
    """Calculates Delta for Black-Scholes."""
    if T <= 0 or sigma == 0: return np.nan # Handle division by zero
    d1_val = d1(S, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(d1_val)
    else:
        return norm.cdf(d1_val) - 1

def gamma(S, K, T, r, sigma):
    """Calculates Gamma for Black-Scholes."""
    if T <= 0 or sigma == 0 or S == 0: return np.nan # Handle division by zero
    d1_val = d1(S, K, T, r, sigma)
    return norm.pdf(d1_val) / (S * sigma * np.sqrt(T))

def theta(S, K, T, r, sigma, option_type="call"):
    """Calculates Theta for Black-Scholes (per year)."""
    if T <= 0 or sigma == 0: return np.nan # Handle division by zero
    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)
    if option_type == "call":
        return (-S * norm.pdf(d1_val) * sigma / (2 * np.sqrt(T)) - 
                r * K * exp(-r * T) * norm.cdf(d2_val))
    else:
        return (-S * norm.pdf(d1_val) * sigma / (2 * np.sqrt(T)) + 
                r * K * exp(-r * T) * norm.cdf(-d2_val))

def vega(S, K, T, r, sigma):
    """Calculates Vega for Black-Scholes."""
    if T <= 0 or sigma == 0: return np.nan # Handle division by zero
    d1_val = d1(S, K, T, r, sigma)
    return S * norm.pdf(d1_val) * np.sqrt(T)

def rho(S, K, T, r, sigma, option_type="call"):
    """Calculates Rho for Black-Scholes."""
    if T <= 0: return np.nan # Cannot calculate Rho for immediate expiry
    d2_val = d2(S, K, T, r, sigma)
    if option_type == "call":
        return K * T * exp(-r * T) * norm.cdf(d2_val)
    else:
        return -K * T * exp(-r * T) * norm.cdf(-d2_val)

# Function to calculate historical volatility
def calculate_historical_volatility(stock_symbol):
    """Calculates annualized historical volatility from yfinance data."""
    try:
        data = yf.download(stock_symbol, period="1y", interval="1d", progress=False)
        # Ensure sufficient data for calculation
        if len(data) < 2:
            st.warning(f"Not enough historical data for {stock_symbol} to calculate volatility.")
            return None
        returns = np.log(data['Adj Close'] / data['Adj Close'].shift(1)).dropna()
        if len(returns) == 0:
            st.warning(f"No valid returns data for {stock_symbol} to calculate volatility.")
            return None
        volatility = returns.std() * np.sqrt(252) # Annualize by sqrt(252 trading days)
        return volatility
    except Exception as e:
        st.error(f"Error calculating historical volatility for {stock_symbol}: {e}")
        return None

# -------------------------
# Streamlit Layout
# -------------------------
st.set_page_config(layout="wide")
st.title("Black-Scholes Options Pricing Model")

# Sidebar for inputs
st.sidebar.markdown("### Connect with me")
st.sidebar.markdown(
    "[📎 LinkedIn](https://www.linkedin.com/in/jonas-f-628179296/)",
    unsafe_allow_html=True
)

st.sidebar.header("Asset Parameters")

input_method = st.sidebar.radio("Select Input Method", ("Search for a Stock", "Enter Manually"))

S, sigma = 100.0, 0.2  # Set default values
ticker = None # Initialize ticker to avoid errors when not in "Search for a Stock" mode

if input_method == "Search for a Stock":
    stock_symbol = st.sidebar.text_input("Enter a stock symbol (e.g., AAPL, GOOGL)", 'AAPL').upper()
    
    if stock_symbol:
        ticker = yf.Ticker(stock_symbol)
        try:
            # Fetch current stock price
            S = ticker.info.get('regularMarketPrice')
            if S is None:
                st.sidebar.error(f"Could not fetch current price for {stock_symbol}. Please try another symbol or enter manually.")
                S = st.sidebar.number_input("Current Asset Price", value=100.0, step=1.0)
            else:
                st.sidebar.markdown(f"**Current Price:** ${S:.2f}")

            # Calculate historical volatility
            sigma = calculate_historical_volatility(stock_symbol)
            if sigma is not None:
                st.sidebar.markdown(f"**Historical Volatility:** {sigma:.2%}")
            else:
                st.sidebar.warning("Could not calculate historical volatility via yfinance. Please enter manually.")
                sigma = st.sidebar.number_input("Manual Volatility (σ)", value=0.2, step=0.01, min_value=0.01)

            # Link to external volatility search
            volatility_link = f"https://www.alphaquery.com/stock/{stock_symbol}/volatility-option-statistics/30-day/historical-volatility"
            st.sidebar.markdown(f"[📊 Search for Historic Volatility]({volatility_link})")

        except (KeyError, IndexError, AttributeError): # Handle ticker.info errors
            st.sidebar.error("Invalid stock symbol or could not fetch data. Please try another symbol or enter manually.")
            S = st.sidebar.number_input("Current Asset Price", value=100.0, step=1.0)
            sigma = st.sidebar.number_input("Volatility (σ)", value=0.2, step=0.01, min_value=0.01)
    else: # If stock_symbol is empty
        S = st.sidebar.number_input("Current Asset Price", value=100.0, step=1.0)
        sigma = st.sidebar.number_input("Volatility (σ)", value=0.2, step=0.01, min_value=0.01)
else:  # "Enter Manually"
    S = st.sidebar.number_input("Current Asset Price", value=100.0, step=1.0)
    sigma = st.sidebar.number_input("Volatility (σ)", value=0.2, step=0.01, min_value=0.01)

st.sidebar.header("Option Parameters")

# --- Select strike strategy or manual input ---
strike_input_method = st.sidebar.radio("Strike Price Input", ("Select Strike Strategy", "Enter Manually"))

K = 100.0 # Default value for K
selected_option_type = "call" # Default option type for single price

if strike_input_method == "Select Strike Strategy":
    if input_method == "Search for a Stock" and ticker:
        try:
            expiries = ticker.options
            if expiries:
                # Use the first available expiry for strike selection
                exp_date_for_strike = expiries[0] 
                chain = ticker.option_chain(exp_date_for_strike)
                
                # Combine and sort all unique strike prices
                all_strikes = sorted(list(set(chain.calls['strike'].tolist() + chain.puts['strike'].tolist())))
                
                # Determine option type for strategy selection
                selected_option_type = st.sidebar.radio("Option Type for Strategy", ["call", "put"], horizontal=True)

                strike_strategy = st.sidebar.selectbox(
                    "Select a Strike Price Strategy",
                    ["At-the-Money (ATM)", "In-the-Money (ITM)", "Out-of-the-Money (OTM)"]
                )

                if strike_strategy == "At-the-Money (ATM)":
                    K = all_strikes[np.abs(np.array(all_strikes) - S).argmin()]
                    st.sidebar.markdown(f"**Selected Strike Price (ATM):** ${K:.2f}")
                elif strike_strategy == "In-the-Money (ITM)":
                    if selected_option_type == "call":
                        itm_strikes = [s for s in all_strikes if s < S]
                        if itm_strikes:
                            K = max(itm_strikes) # Highest strike below S
                            st.sidebar.markdown(f"**Selected Strike Price (ITM Call):** ${K:.2f}")
                        else:
                            st.sidebar.warning("No ITM Call options found. Using ATM strike.")
                            K = all_strikes[np.abs(np.array(all_strikes) - S).argmin()]
                            st.sidebar.markdown(f"**Selected Strike Price (ATM):** ${K:.2f}")
                    else: # put
                        itm_strikes = [s for s in all_strikes if s > S]
                        if itm_strikes:
                            K = min(itm_strikes) # Lowest strike above S
                            st.sidebar.markdown(f"**Selected Strike Price (ITM Put):** ${K:.2f}")
                        else:
                            st.sidebar.warning("No ITM Put options found. Using ATM strike.")
                            K = all_strikes[np.abs(np.array(all_strikes) - S).argmin()]
                            st.sidebar.markdown(f"**Selected Strike Price (ATM):** ${K:.2f}")

                elif strike_strategy == "Out-of-the-Money (OTM)":
                    if selected_option_type == "call":
                        otm_strikes = [s for s in all_strikes if s > S]
                        if otm_strikes:
                            K = min(otm_strikes) # Lowest strike above S
                            st.sidebar.markdown(f"**Selected Strike Price (OTM Call):** ${K:.2f}")
                        else:
                            st.sidebar.warning("No OTM Call options found. Using ATM strike.")
                            K = all_strikes[np.abs(np.array(all_strikes) - S).argmin()]
                            st.sidebar.markdown(f"**Selected Strike Price (ATM):** ${K:.2f}")
                    else: # put
                        otm_strikes = [s for s in all_strikes if s < S]
                        if otm_strikes:
                            K = max(otm_strikes) # Highest strike below S
                            st.sidebar.markdown(f"**Selected Strike Price (OTM Put):** ${K:.2f}")
                        else:
                            st.sidebar.warning("No OTM Put options found. Using ATM strike.")
                            K = all_strikes[np.abs(np.array(all_strikes) - S).argmin()]
                            st.sidebar.markdown(f"**Selected Strike Price (ATM):** ${K:.2f}")
            else:
                st.sidebar.warning("No option data available for this stock. Please enter strike manually.")
                K = st.sidebar.number_input("Manual Strike Price", value=100.0, step=1.0)
                selected_option_type = st.sidebar.radio("Option Type", ["call", "put"], horizontal=True)
        except Exception as e:
            st.sidebar.warning(f"Error fetching strike prices: {e}. Please enter manually.")
            K = st.sidebar.number_input("Manual Strike Price", value=100.0, step=1.0)
            selected_option_type = st.sidebar.radio("Option Type", ["call", "put"], horizontal=True)
    else:
        st.sidebar.warning("Please search for a stock to use this feature.")
        K = st.sidebar.number_input("Manual Strike Price", value=100.0, step=1.0)
        selected_option_type = st.sidebar.radio("Option Type", ["call", "put"], horizontal=True)

else: # "Enter Manually"
    K = st.sidebar.number_input("Manual Strike Price", value=100.0, step=1.0)
    selected_option_type = st.sidebar.radio("Option Type", ["call", "put"], horizontal=True)


T = st.sidebar.number_input("Time to Maturity (Years)", value=1.0, step=0.1, min_value=0.01)
r = st.sidebar.number_input("Risk-Free Interest Rate", value=0.05, step=0.01)

# -------------------------
# Tabs for Organization
# -------------------------
tab1, tab2, tab3 = st.tabs(["Black-Scholes Calculator", "Option Greeks", "Volatility & P&L Analysis"])

with tab1:
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
    S_min = st.sidebar.number_input("Min Spot Price", value=default_s_min, step=1.0, help="Minimum stock price for the heatmap range.")
    S_max = st.sidebar.number_input("Max Spot Price", value=default_s_max, step=1.0, help="Maximum stock price for the heatmap range.")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Volatility Range for Heatmap")
    sigma_min = st.sidebar.slider("Min Volatility", 0.01, 1.0, 0.1, step=0.01, help="Minimum volatility for the heatmap range.")
    sigma_max = st.sidebar.slider("Max Volatility", 0.01, 1.0, 0.3, step=0.01, help="Maximum volatility for the heatmap range.")
    
    spot_prices_hm = np.linspace(S_min, S_max, 10)
    vols_hm = np.linspace(sigma_min, sigma_max, 9)
    call_prices_hm = np.zeros((len(vols_hm), len(spot_prices_hm)))
    put_prices_hm = np.zeros((len(vols_hm), len(spot_prices_hm)))
    for i, v in enumerate(vols_hm):
        for j, s_val in enumerate(spot_prices_hm):
            if s_val > 0:
                call_prices_hm[i, j] = bs_price(s_val, K, T, r, v, "call")
                put_prices_hm[i, j] = bs_price(s_val, K, T, r, v, "put")
    call_min_price_hm, call_max_price_hm = call_prices_hm.min(), call_prices_hm.max()
    put_min_price_hm, put_max_price_hm = put_prices_hm.min(), put_prices_hm.max()

    def create_heatmap(prices, x_labels, y_labels, title, min_price, max_price):
        custom_colorscale = [
            [0.0, 'rgb(180, 219, 203)'], # Lighter Green
            [0.2, 'rgb(102, 184, 159)'], # Green
            [0.4, 'rgb(240, 240, 240)'], # Light Grey
            [0.6, 'rgb(255, 150, 150)'], # Light Red
            [0.8, 'rgb(255, 100, 100)'], # Red
            [1.0, 'rgb(255, 50, 50)']    # Dark Red
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
        # Calculate a robust mid_point for text color contrast
        mid_point = np.percentile(prices, 50) 
        for i, vol_label in enumerate(y_labels):
            for j, spot_label in enumerate(x_labels):
                annotations.append(
                    dict(
                        x=spot_label,
                        y=vol_label,
                        text=f"{prices[i, j]:.2f}",
                        showarrow=False,
                        font=dict(color="white" if prices[i,j] > mid_point else "black", size=8) # Smaller font for annotations
                    )
                )
        fig.update_layout(
            title=title,
            xaxis=dict(title="Spot Price", tickmode='array', tickvals=x_labels, ticktext=[f'{p:.2f}' for p in x_labels]),
            yaxis=dict(title="Volatility", tickmode='array', tickvals=y_labels, ticktext=[f'{v:.2f}' for v in y_labels]),
            annotations=annotations,
            height=500 # Adjust height for better visibility
        )
        return fig

    # -------------------------
    # Create and display heatmaps
    # -------------------------
    col3, col4 = st.columns(2)
    with col3:
        fig_call_hm = create_heatmap(call_prices_hm, spot_prices_hm, vols_hm, "CALL Option Price Heatmap", call_min_price_hm, call_max_price_hm)
        st.plotly_chart(fig_call_hm, use_container_width=True)
    with col4:
        fig_put_hm = create_heatmap(put_prices_hm, spot_prices_hm, vols_hm, "PUT Option Price Heatmap", put_min_price_hm, put_max_price_hm)
        st.plotly_chart(fig_put_hm, use_container_width=True)

    # -------------------------
    # Options Screener
    # -------------------------
    st.subheader("Highest Potential Profit Screener")
    if input_method == "Search for a Stock" and stock_symbol and S is not None:
        target_price = st.number_input(
            "Your **Target Stock Price** for Screener", 
            value=float(S), 
            step=1.0, 
            help="Enter the price you believe the stock will reach by the nearest expiry. The screener will find options that could be most profitable if the stock reaches this price."
        )

        try:
            expiries = ticker.options
            if not expiries:
                st.warning("No option chain data available for this stock for screener.")
            else:
                exp_date_screener = expiries[0]
                st.info(f"Analyzing options for nearest expiry: **{exp_date_screener}**")

                chain = yf.Ticker(stock_symbol).option_chain(exp_date_screener)
                calls_screener = chain.calls
                puts_screener = chain.puts
                
                profitable_options = []
                
                # Time to expiry in years for screener
                days_to_expiry_screener = (pd.to_datetime(exp_date_screener) - pd.Timestamp.now()).days
                if days_to_expiry_screener <= 0:
                    st.warning("Expiration date for screener is in the past or today. Cannot calculate profit.")
                else:
                    T_days_screener = days_to_expiry_screener / 365.0

                    # Analyze call options for screener
                    for _, row in calls_screener.iterrows():
                        market_price = row.get('lastPrice', row.get('bid'))
                        strike = row['strike']

                        if not pd.isna(market_price) and market_price > 0:
                            theoretical_value_at_target = bs_price(target_price, strike, T_days_screener, r, sigma, 'call')
                            potential_profit = theoretical_value_at_target - market_price
                            
                            if potential_profit > 0 and market_price > 0: # Ensure market_price is not zero for ROI
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

                    # Analyze put options for screener
                    for _, row in puts_screener.iterrows():
                        market_price = row.get('lastPrice', row.get('bid'))
                        strike = row['strike']

                        if not pd.isna(market_price) and market_price > 0:
                            theoretical_value_at_target = bs_price(target_price, strike, T_days_screener, r, sigma, 'put')
                            potential_profit = theoretical_value_at_target - market_price
                            
                            if potential_profit > 0 and market_price > 0: # Ensure market_price is not zero for ROI
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
                        df_profitable = df_profitable.drop(columns=['sort_roi'])
                        st.markdown(df_profitable.to_html(escape=False), unsafe_allow_html=True)
                    else:
                        st.info("No options with a positive potential profit found based on your target price for this screener.")
        
        except Exception as e:
            st.error(f"Error fetching option chain data for screener: {e}")
    else:
        st.info("Please search for a stock to use the profit screener.")


with tab2:
    st.header("Option Greeks Analysis")
    st.markdown("Here you can analyze the sensitivity of the currently selected option to various market factors. These values are based on the **Spot Price**, **Strike Price**, **Time to Maturity**, **Risk-Free Rate**, and **Volatility** set in the sidebar.")

    if S is not None and K is not None and T > 0 and r is not None and sigma is not None:
        st.subheader(f"Greeks for {selected_option_type.upper()} Option (S=${S:.2f}, K=${K:.2f}, T={T:.2f}yr, r={r:.2%}, σ={sigma:.2%})")
        
        # Custom CSS for the colored boxes
        st.markdown("""
            <style>
            .metric-box {
                border-radius: 5px;
                padding: 10px;
                margin-bottom: 10px;
                text-align: center;
                font-weight: bold;
            }
            .green-box {
                background-color: #d4edda;
                color: #155724;
            }
            .red-box {
                background-color: #f8d7da;
                color: #721c24;
            }
            .neutral-box {
                background-color: #e2e3e5;
                color: #383d41;
            }
            .metric-value {
                font-size: 1.5em;
                margin-bottom: 5px;
            }
            .metric-label {
                font-size: 0.9em;
                color: #555;
            }
            .metric-interpretation {
                font-size: 0.8em;
                margin-top: 5px;
                text-align: left;
            }
            </style>
        """, unsafe_allow_html=True)

        col_delta, col_gamma, col_theta, col_vega, col_rho = st.columns(5)
        
        # --- Delta ---
        with col_delta:
            option_delta = delta(S, K, T, r, sigma, selected_option_type)
            box_class = "neutral-box"
            if not np.isnan(option_delta):
                if selected_option_type == "call":
                    box_class = "green-box" if option_delta > 0 else "red-box"
                else: # put
                    box_class = "green-box" if option_delta < 0 else "red-box"
            
            st.markdown(f"""
            <div class="metric-box {box_class}">
                <div class="metric-label">Delta (Δ)</div>
                <div class="metric-value">{option_delta:.3f}</div>
                <div class="metric-interpretation">
                    **Was es bedeutet:** Erwartete Veränderung des Optionspreises bei einer ${S_min} Kursänderung der Aktie.
                    Eine {selected_option_type}-Option ändert ihren Preis voraussichtlich um ${abs(option_delta):.2f}.
                    {"" if np.isnan(option_delta) else f"Deine {selected_option_type}-Option wird {'' if option_delta > 0 else 'weniger '} wertvoller, wenn der Aktienkurs steigt." if selected_option_type == 'call' else f"Deine {selected_option_type}-Option wird {'' if option_delta < 0 else 'weniger '} wertvoller, wenn der Aktienkurs fällt."}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # --- Gamma ---
        with col_gamma:
            option_gamma = gamma(S, K, T, r, sigma)
            st.markdown(f"""
            <div class="metric-box neutral-box">
                <div class="metric-label">Gamma (Γ)</div>
                <div class="metric-value">{option_gamma:.3f}</div>
                <div class="metric-interpretation">
                    **Was es bedeutet:** Misst, wie stark sich das Delta ändert, wenn sich der Aktienkurs um $1 bewegt.
                    Ein hohes Gamma bedeutet, dass das Delta (und damit der Optionspreis) sehr empfindlich auf Aktienkursänderungen reagiert.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # --- Theta ---
        with col_theta:
            option_theta = theta(S, K, T, r, sigma, selected_option_type)
            box_class = "neutral-box"
            if not np.isnan(option_theta):
                box_class = "red-box" # Time decay is generally a cost for long options
            
            st.markdown(f"""
            <div class="metric-box {box_class}">
                <div class="metric-label">Theta (Θ)</div>
                <div class="metric-value">{option_theta:.3f} (pro Jahr)</div>
                <div class="metric-interpretation">
                    **Was es bedeutet:** Zeitwertverlust. Der Optionspreis verringert sich voraussichtlich um ${abs(option_theta / 365):.3f} pro Tag.
                    Dies ist der "Kostenfaktor" des Zeitablaufs für eine Long-Option.
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # --- Vega ---
        with col_vega:
            option_vega = vega(S, K, T, r, sigma)
            box_class = "neutral-box"
            if not np.isnan(option_vega):
                box_class = "green-box" if option_vega > 0 else "red-box" # Long options benefit from rising vol
            
            st.markdown(f"""
            <div class="metric-box {box_class}">
                <div class="metric-label">Vega (ν)</div>
                <div class="metric-value">{option_vega:.3f}</div>
                <div class="metric-interpretation">
                    **Was es bedeutet:** Sensitivität gegenüber Volatilität. Eine 1%ige (0.01) Erhöhung der Volatilität würde den Optionspreis um ${option_vega / 100:.2f} verändern.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # --- Rho ---
        with col_rho:
            option_rho = rho(S, K, T, r, sigma, selected_option_type)
            box_class = "neutral-box"
            if not np.isnan(option_rho):
                if selected_option_type == "call":
                    box_class = "green-box" if option_rho > 0 else "red-box"
                else: # put
                    box_class = "green-box" if option_rho < 0 else "red-box"
            
            st.markdown(f"""
            <div class="metric-box {box_class}">
                <div class="metric-label">Rho (ρ)</div>
                <div class="metric-value">{option_rho:.3f}</div>
                <div class="metric-interpretation">
                    **Was es bedeutet:** Sensitivität gegenüber Zinsen. Eine 1%ige (0.01) Erhöhung des risikofreien Zinssatzes würde den Optionspreis um ${option_rho / 100:.2f} verändern.
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Bitte stelle sicher, dass alle 'Asset Parameters' und 'Option Parameters' im Tab 'Black-Scholes Calculator' korrekt eingestellt sind, um die Griechen anzuzeigen.")


with tab3:
    st.header("Volatility & P&L Analysis")
    st.markdown("Hier kannst du untersuchen, wie sich die implizite Volatilität über verschiedene Ausübungspreise verhält und potenzielle Gewinn-/Verlustszenarien visualisieren.")

    if input_method == "Search for a Stock" and stock_symbol and S is not None:
        st.subheader("Volatility Smile/Skew")
        st.write("Der **Volatility Smile** (oder Volatility Skew) zeigt die implizite Volatilität (IV) verschiedener Optionen des gleichen Basiswerts mit der gleichen Laufzeit, aber unterschiedlichen Ausübungspreisen. Statt einer geraden Linie (wie das Black-Scholes-Modell annimmt), siehst du oft eine U-förmige Kurve, die die Markterwartung über zukünftige Kursbewegungen widerspiegelt.")
        st.info("**Für Anfänger:** Ein 'Smile' bedeutet, dass der Markt davon ausgeht, dass extremere Preisbewegungen (Optionen weit im oder aus dem Geld) wahrscheinlicher sind und daher eine höhere implizite Volatilität aufweisen.")

        try:
            expiries = ticker.options
            if expiries:
                # Use the first expiry for volatility smile
                selected_smile_exp_date = st.selectbox("1. Wähle ein Verfallsdatum für den Volatility Smile", expiries, help="Wähle das Verfallsdatum, für das du die implizite Volatilität über verschiedene Ausübungspreise hinweg sehen möchtest.")

                chain = ticker.option_chain(selected_smile_exp_date)
                calls_smile = chain.calls
                puts_smile = chain.puts

                vol_smile_data = []

                # Calculate IV for calls
                for _, row in calls_smile.iterrows():
                    market_price = row.get('lastPrice', row.get('bid'))
                    strike = row['strike']
                    
                    days_to_expiry_smile = (pd.to_datetime(selected_smile_exp_date) - pd.Timestamp.now()).days
                    T_smile = days_to_expiry_smile / 365.0 if days_to_expiry_smile > 0 else 0.01 # Avoid T=0

                    if not pd.isna(market_price) and market_price > 0 and T_smile > 0:
                        iv = find_implied_volatility(market_price, S, strike, T_smile, r, 'call')
                        if not np.isnan(iv):
                            vol_smile_data.append({'Strike': strike, 'Implied Volatility': iv, 'Option Type': 'Call'})
                
                # Calculate IV for puts
                for _, row in puts_smile.iterrows():
                    market_price = row.get('lastPrice', row.get('bid'))
                    strike = row['strike']

                    days_to_expiry_smile = (pd.to_datetime(selected_smile_exp_date) - pd.Timestamp.now()).days
                    T_smile = days_to_expiry_smile / 365.0 if days_to_expiry_smile > 0 else 0.01 # Avoid T=0

                    if not pd.isna(market_price) and market_price > 0 and T_smile > 0:
                        iv = find_implied_volatility(market_price, S, strike, T_smile, r, 'put')
                        if not np.isnan(iv):
                            vol_smile_data.append({'Strike': strike, 'Implied Volatility': iv, 'Option Type': 'Put'})
                
                if vol_smile_data:
                    df_smile = pd.DataFrame(vol_smile_data)
                    fig_smile = go.Figure()
                    
                    # Plot Calls
                    fig_smile.add_trace(go.Scatter(
                        x=df_smile[df_smile['Option Type'] == 'Call']['Strike'],
                        y=df_smile[df_smile['Option Type'] == 'Call']['Implied Volatility'],
                        mode='lines+markers',
                        name='Call Optionen',
                        line=dict(color='blue')
                    ))
                    # Plot Puts
                    fig_smile.add_trace(go.Scatter(
                        x=df_smile[df_smile['Option Type'] == 'Put']['Strike'],
                        y=df_smile[df_smile['Option Type'] == 'Put']['Implied Volatility'],
                        mode='lines+markers',
                        name='Put Optionen',
                        line=dict(color='red')
                    ))
                    
                    fig_smile.update_layout(
                        title=f"Volatility Smile/Skew für {stock_symbol} ({selected_smile_exp_date})",
                        xaxis_title="Ausübungspreis (Strike Price)",
                        yaxis_title="Implizite Volatilität",
                        hovermode="x unified",
                        height=500
                    )
                    st.plotly_chart(fig_smile, use_container_width=True)
                else:
                    st.info("Konnte den Volatility Smile nicht generieren. Es wurden keine gültigen Optionsdaten mit impliziter Volatilität gefunden.")
            else:
                st.info("Für den Volatility Smile sind keine Optionskettendaten verfügbar.")
        except Exception as e:
            st.error(f"Fehler beim Generieren des Volatility Smile: {e}")
    else:
        st.info("Bitte suche einen Aktienkurs im Tab 'Black-Scholes Calculator', um den Volatility Smile anzuzeigen.")

    st.markdown("---")
    st.subheader("Interaktive P&L (Profit & Loss) Charts & Greeks Plots")
    st.write("Hier kannst du für eine ausgewählte Option deren potenziellen Gewinn/Verlust am Verfallstag und die Entwicklung der Griechen bei unterschiedlichen Aktienkursen analysieren.")
    st.info("**Für Anfänger:** Das P&L-Diagramm zeigt, wie viel Geld du mit deiner Option verdienen oder verlieren könntest, je nachdem, wo der Aktienkurs am Verfallstag steht. Die Griechen-Diagramme zeigen, wie empfindlich der Wert deiner Option auf Veränderungen des Aktienkurses reagiert.")

    if input_method == "Search for a Stock" and stock_symbol and S is not None:
        try:
            expiries_pnl = ticker.options
            if expiries_pnl:
                selected_pnl_exp_date = st.selectbox("2. Wähle ein Verfallsdatum für P&L und Griechen-Diagramme", expiries_pnl, help="Wähle das Verfallsdatum für die Analyse des Gewinn-/Verlustprofils und der Sensitivität der Option.")
                chain_pnl = ticker.option_chain(selected_pnl_exp_date)
                
                # Combine calls and puts for selection
                all_options_pnl = pd.concat([chain_pnl.calls, chain_pnl.puts])
                
                # Create display string for selectbox
                all_options_pnl['display_name'] = all_options_pnl.apply(
                    lambda row: f"{'Call' if row['contractSymbol'].endswith('C') else 'Put'} - Strike: {row['strike']:.2f} - Last Price: {row.get('lastPrice', row.get('bid')):.2f}",
                    axis=1
                )
                
                selected_option_pnl_str = st.selectbox(
                    "3. Wähle eine Option zur Analyse (P&L und Griechen-Diagramme)", 
                    all_options_pnl['display_name'].tolist(),
                    help="Wähle eine spezifische Option aus der Optionskette aus, um ihre Gewinn-/Verlust-Potenziale und ihre Griechen-Werte zu visualisieren."
                )

                if selected_option_pnl_str:
                    # Extract the original row for the selected option
                    selected_option_row = all_options_pnl[all_options_pnl['display_name'] == selected_option_pnl_str].iloc[0]
                    
                    pnl_option_type = 'call' if selected_option_row['contractSymbol'].endswith('C') else 'put'
                    pnl_strike = selected_option_row['strike']
                    pnl_market_price = selected_option_row.get('lastPrice', selected_option_row.get('bid'))

                    days_to_expiry_pnl = (pd.to_datetime(selected_pnl_exp_date) - pd.Timestamp.now()).days
                    T_pnl = days_to_expiry_pnl / 365.0 if days_to_expiry_pnl > 0 else 0.01 # Ensure T_pnl is positive for calculations

                    # --- P&L Chart Calculation ---
                    # Range for stock price at expiration
                    spot_range = np.linspace(max(1, S - 0.5 * S), S + 0.5 * S, 200) # Extend range +/- 50% of current spot
                    pnl_values = []
                    for spot_price_at_expiry in spot_range:
                        # Profit/Loss is (Value at Expiry - Purchase Price)
                        if pnl_option_type == 'call':
                            value_at_expiry = max(0, spot_price_at_expiry - pnl_strike)
                        else: # put
                            value_at_expiry = max(0, pnl_strike - spot_price_at_expiry)
                        pnl_values.append(value_at_expiry - pnl_market_price)

                    fig_pnl = go.Figure()
                    fig_pnl.add_trace(go.Scatter(x=spot_range, y=pnl_values, mode='lines', name='P&L am Verfallstag'))
                    fig_pnl.add_hline(y=0, line_dash="dash", line_color="grey", annotation_text="Break-Even", annotation_position="bottom right")
                    fig_pnl.add_vline(x=S, line_dash="dot", line_color="orange", annotation_text="Aktueller Kurs", annotation_position="top right")
                    fig_pnl.update_layout(
                        title=f"Gewinn-/Verlustprofil für ausgewählte {pnl_option_type.upper()}-Option am Verfallstag",
                        xaxis_title="Aktienkurs am Verfallstag",
                        yaxis_title="Gewinn/Verlust",
                        height=500
                    )
                    st.plotly_chart(fig_pnl, use_container_width=True)

                    st.markdown("---")
                    st.subheader("Greeks Plots vs. Stock Price")
                    st.write("Diese Diagramme zeigen, wie sich die Werte der Optionen (Delta, Gamma, Theta, Vega) verändern, wenn der Aktienkurs steigt oder fällt, unter Beibehaltung der anderen Parameter. Dies hilft dir, die Sensitivität deiner Option besser zu verstehen.")

                    # --- Greeks Plots Calculation ---
                    greeks_spot_range = np.linspace(max(1, S - 0.2 * S), S + 0.2 * S, 50) # Smaller range for Greeks sensitivity, +/- 20%
                    
                    # Ensure T_pnl and sigma are valid for greek calculations
                    if T_pnl > 0 and sigma > 0:
                        deltas = [delta(s_val, pnl_strike, T_pnl, r, sigma, pnl_option_type) for s_val in greeks_spot_range]
                        gammas = [gamma(s_val, pnl_strike, T_pnl, r, sigma) for s_val in greeks_spot_range]
                        thetas = [theta(s_val, pnl_strike, T_pnl, r, sigma, pnl_option_type) for s_val in greeks_spot_range]
                        vegas = [vega(s_val, pnl_strike, T_pnl, r, sigma) for s_val in greeks_spot_range]

                        fig_greeks = go.Figure()
                        fig_greeks.add_trace(go.Scatter(x=greeks_spot_range, y=deltas, mode='lines', name='Delta'))
                        fig_greeks.add_trace(go.Scatter(x=greeks_spot_range, y=gammas, mode='lines', name='Gamma'))
                        fig_greeks.add_trace(go.Scatter(x=greeks_spot_range, y=thetas, mode='lines', name='Theta (jährl.)'))
                        fig_greeks.add_trace(go.Scatter(x=greeks_spot_range, y=vegas, mode='lines', name='Vega'))

                        fig_greeks.update_layout(
                            title=f"Options-Griechen vs. Aktienkurs für ausgewählte {pnl_option_type.upper()}-Option",
                            xaxis_title="Aktienkurs",
                            yaxis_title="Wert des Griechen",
                            hovermode="x unified",
                            height=600
                        )
                        st.plotly_chart(fig_greeks, use_container_width=True)
                    else:
                        st.warning("Kann Griechen-Diagramme nicht berechnen: Zeit bis zur Fälligkeit oder Volatilität ist Null.")

                else:
                    st.info("Bitte wähle eine Option aus, um P&L und Griechen-Diagramme anzuzeigen.")
            else:
                st.info("Für P&L und Griechen-Diagramme sind keine Optionskettendaten verfügbar.")
        except Exception as e:
            st.error(f"Fehler beim Generieren des P&L-Charts oder der Griechen-Diagramme: {e}")
    else:
        st.info("Bitte suche einen Aktienkurs im Tab 'Black-Scholes Calculator', um P&L und Griechen-Analyse anzuzeigen.")
