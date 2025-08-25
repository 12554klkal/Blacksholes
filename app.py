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
tab1, tab2, tab3, tab4 = st.tabs(["Black-Scholes Calculator", "Option Greeks", "Volatility & P&L Analysis", "Financial Terms Explained"])

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
            .metric-box-v2 {
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 15px;
                text-align: left;
                font-weight: bold;
                box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
            }
            .green-box-v2 {
                background-color: #d4edda;
                color: #155724;
            }
            .red-box-v2 {
                background-color: #f8d7da;
                color: #721c24;
            }
            .neutral-box-v2 {
                background-color: #e2e3e5;
                color: #383d41;
            }
            .metric-title-v2 {
                font-size: 1.2em;
                margin-bottom: 5px;
                color: #212529;
            }
            .metric-value-v2 {
                font-size: 1.8em;
                color: #000;
                margin-bottom: 10px;
            }
            .metric-interpretation-v2 {
                font-size: 0.9em;
                color: #495057;
                line-height: 1.4;
            }
            </style>
        """, unsafe_allow_html=True)

        col_delta, col_gamma, col_theta, col_vega, col_rho = st.columns(5)
        
        # --- Delta ---
        with col_delta:
            option_delta = delta(S, K, T, r, sigma, selected_option_type)
            box_class = "neutral-box-v2"
            interpretation_text = ""
            if not np.isnan(option_delta):
                if selected_option_type == "call":
                    box_class = "green-box-v2" if option_delta > 0 else "red-box-v2"
                    interpretation_text = f"Your Call option's price is expected to change by ${abs(option_delta):.2f} for a $1 move in the stock. A positive Delta means it generally gains as the stock price rises."
                else: # put
                    box_class = "green-box-v2" if option_delta < 0 else "red-box-v2"
                    interpretation_text = f"Your Put option's price is expected to change by ${abs(option_delta):.2f} for a $1 move in the stock. A negative Delta means it generally gains as the stock price falls."
            
            st.markdown(f"""
            <div class="metric-box-v2 {box_class}">
                <div class="metric-title-v2">Delta (Δ)</div>
                <div class="metric-value-v2">{option_delta:.3f}</div>
                <div class="metric-interpretation-v2">
                    **Meaning:** Measures how much the option price is expected to change for every $1 change in the underlying stock's price.<br>
                    {interpretation_text}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # --- Gamma ---
        with col_gamma:
            option_gamma = gamma(S, K, T, r, sigma)
            st.markdown(f"""
            <div class="metric-box-v2 neutral-box-v2">
                <div class="metric-title-v2">Gamma (Γ)</div>
                <div class="metric-value-v2">{option_gamma:.3f}</div>
                <div class="metric-interpretation-v2">
                    **Meaning:** Measures the rate of change of Delta. A high Gamma means your Delta (and thus your option's sensitivity) will change rapidly with small movements in the stock price.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # --- Theta ---
        with col_theta:
            option_theta = theta(S, K, T, r, sigma, selected_option_type)
            box_class = "red-box-v2" if not np.isnan(option_theta) else "neutral-box-v2" # Time decay is a cost for long options
            
            st.markdown(f"""
            <div class="metric-box-v2 {box_class}">
                <div class="metric-title-v2">Theta (Θ)</div>
                <div class="metric-value-v2">{option_theta:.3f} (per year)</div>
                <div class="metric-interpretation-v2">
                    **Meaning:** Represents time decay. The option price is expected to decrease by ${abs(option_theta / 365):.3f} per day due to the passage of time.
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # --- Vega ---
        with col_vega:
            option_vega = vega(S, K, T, r, sigma)
            box_class = "green-box-v2" if (not np.isnan(option_vega) and option_vega > 0) else "red-box-v2" if (not np.isnan(option_vega) and option_vega < 0) else "neutral-box-v2" # Long options benefit from rising vol
            
            st.markdown(f"""
            <div class="metric-box-v2 {box_class}">
                <div class="metric-title-v2">Vega (ν)</div>
                <div class="metric-value-v2">{option_vega:.3f}</div>
                <div class="metric-interpretation-v2">
                    **Meaning:** Measures the option's sensitivity to a 1% (0.01) change in **implied volatility**. A 1% rise in volatility would change the option price by ${option_vega / 100:.2f}.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # --- Rho ---
        with col_rho:
            option_rho = rho(S, K, T, r, sigma, selected_option_type)
            box_class = "neutral-box-v2"
            if not np.isnan(option_rho):
                if selected_option_type == "call":
                    box_class = "green-box-v2" if option_rho > 0 else "red-box-v2"
                    interpretation_text = "Call options generally benefit from higher interest rates."
                else: # put
                    box_class = "green-box-v2" if option_rho < 0 else "red-box-v2"
                    interpretation_text = "Put options generally suffer from higher interest rates."
            
            st.markdown(f"""
            <div class="metric-box-v2 {box_class}">
                <div class="metric-title-v2">Rho (ρ)</div>
                <div class="metric-value-v2">{option_rho:.3f}</div>
                <div class="metric-interpretation-v2">
                    **Meaning:** Measures the option's sensitivity to a 1% (0.01) change in the **risk-free interest rate**. A 1% rise in interest rates would change the option price by ${option_rho / 100:.2f}.<br>
                    {interpretation_text}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Please ensure all 'Asset Parameters' and 'Option Parameters' are set correctly on the 'Black-Scholes Calculator' tab to view Greeks.")


with tab3:
    st.header("Volatility & P&L Analysis")
    st.markdown("Here you can explore how implied volatility behaves across different strike prices and visualize potential profit/loss scenarios for specific options.")

    if input_method == "Search for a Stock" and stock_symbol and S is not None:
        st.subheader("Volatility Smile/Skew")
        st.write("The **Volatility Smile** (or Volatility Skew) plots the **implied volatility (IV)** of different options for the same underlying asset and expiration date, but with varying strike prices. Instead of a flat line (as assumed by the basic Black-Scholes model), you often see a U-shaped curve, which reflects market expectations about future price movements.")
        st.info("**For Beginners:** A 'smile' indicates that the market expects more extreme price movements (for options far In-the-Money or Out-of-the-Money) to be more likely, thus assigning them a higher implied volatility.")

        try:
            expiries = ticker.options
            if expiries:
                selected_smile_exp_date = st.selectbox(
                    "1. Select Expiration Date for Volatility Smile", 
                    expiries, 
                    help="Choose the expiration date for which you want to see the implied volatility across different strike prices."
                )

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
                    df_calls = df_smile[df_smile['Option Type'] == 'Call']
                    if not df_calls.empty:
                        fig_smile.add_trace(go.Scatter(
                            x=df_calls['Strike'],
                            y=df_calls['Implied Volatility'],
                            mode='lines+markers',
                            name='Call Options',
                            line=dict(color='blue')
                        ))
                    
                    # Plot Puts
                    df_puts = df_smile[df_smile['Option Type'] == 'Put']
                    if not df_puts.empty:
                        fig_smile.add_trace(go.Scatter(
                            x=df_puts['Strike'],
                            y=df_puts['Implied Volatility'],
                            mode='lines+markers',
                            name='Put Options',
                            line=dict(color='red')
                        ))
                    
                    fig_smile.update_layout(
                        title=f"Volatility Smile/Skew for {stock_symbol} ({selected_smile_exp_date})",
                        xaxis_title="Strike Price",
                        yaxis_title="Implied Volatility",
                        hovermode="x unified",
                        height=500
                    )
                    st.plotly_chart(fig_smile, use_container_width=True)
                else:
                    st.info("Could not generate Volatility Smile. No valid option data with implied volatility found for the selected expiry.")
            else:
                st.info("No option chain data available for volatility smile.")
        except Exception as e:
            st.error(f"Error generating Volatility Smile: {e}")
    else:
        st.info("Please search for a stock on the 'Black-Scholes Calculator' tab to view Volatility Smile.")

    st.markdown("---")
    st.subheader("Interactive P&L (Profit & Loss) Chart & Greeks Plots")
    st.write("Here you can analyze the potential **Profit/Loss (P&L)** of a selected option at expiration, and see how its **Greeks** change with different stock prices.")
    st.info("**For Beginners:** The P&L chart shows how much money you could gain or lose with your option, depending on the stock price at expiration. The Greeks plots show how sensitive your option's value is to changes in the stock price.")

    if input_method == "Search for a Stock" and stock_symbol and S is not None:
        try:
            expiries_pnl = ticker.options
            if expiries_pnl:
                selected_pnl_exp_date = st.selectbox(
                    "2. Select Expiration Date for P&L and Greeks Plots", 
                    expiries_pnl, 
                    help="Choose the expiration date for analyzing the profit/loss profile and sensitivity of an option."
                )
                chain_pnl = ticker.option_chain(selected_pnl_exp_date)
                
                # Combine calls and puts for selection
                all_options_pnl = pd.concat([chain_pnl.calls, chain_pnl.puts])
                
                # Create display string for selectbox
                all_options_pnl['display_name'] = all_options_pnl.apply(
                    lambda row: f"{'Call' if row['contractSymbol'].endswith('C') else 'Put'} - Strike: {row['strike']:.2f} - Last Price: {row.get('lastPrice', row.get('bid')):.2f}",
                    axis=1
                )
                
                selected_option_pnl_str = st.selectbox(
                    "3. Select an Option to Analyze (P&L and Greeks Plots)", 
                    all_options_pnl['display_name'].tolist(),
                    help="Choose a specific option from the option chain to visualize its profit/loss potential and Greek values."
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
                    # Ensure S is valid before using it for range calculation
                    if S is not None and S > 0:
                        spot_range = np.linspace(max(1, S - 0.5 * S), S + 0.5 * S, 200) # Extend range +/- 50% of current spot
                    else:
                        spot_range = np.linspace(50, 150, 200) # Fallback range
                        st.warning("Current stock price (S) is not available or invalid; using default range for P&L chart.")

                    pnl_values = []
                    for spot_price_at_expiry in spot_range:
                        # Profit/Loss is (Value at Expiry - Purchase Price)
                        if pnl_option_type == 'call':
                            value_at_expiry = max(0, spot_price_at_expiry - pnl_strike)
                        else: # put
                            value_at_expiry = max(0, pnl_strike - spot_price_at_expiry)
                        pnl_values.append(value_at_expiry - pnl_market_price)

                    fig_pnl = go.Figure()
                    fig_pnl.add_trace(go.Scatter(x=spot_range, y=pnl_values, mode='lines', name='P&L at Expiry'))
                    fig_pnl.add_hline(y=0, line_dash="dash", line_color="grey", annotation_text="Break-Even", annotation_position="bottom right")
                    if S is not None and S > 0:
                        fig_pnl.add_vline(x=S, line_dash="dot", line_color="orange", annotation_text="Current Spot Price", annotation_position="top right")
                    fig_pnl.update_layout(
                        title=f"Profit/Loss Profile for Selected {pnl_option_type.upper()} Option at Expiry",
                        xaxis_title="Stock Price at Expiration",
                        yaxis_title="Profit/Loss",
                        height=500
                    )
                    st.plotly_chart(fig_pnl, use_container_width=True)

                    st.markdown("---")
                    st.subheader("Greeks Plots vs. Stock Price")
                    st.write("These charts show how the option's Greek values (Delta, Gamma, Theta, Vega) change as the stock price fluctuates, keeping other parameters constant. This helps you understand your option's sensitivity under different market conditions.")

                    # --- Greeks Plots Calculation ---
                    # Ensure S is valid before using it for range calculation
                    if S is not None and S > 0:
                        greeks_spot_range = np.linspace(max(1, S - 0.2 * S), S + 0.2 * S, 50) # Smaller range for Greeks sensitivity, +/- 20%
                    else:
                        greeks_spot_range = np.linspace(50, 150, 50) # Fallback range
                        st.warning("Current stock price (S) is not available or invalid; using default range for Greeks plots.")
                    
                    # Ensure T_pnl and sigma are valid for greek calculations
                    if T_pnl > 0 and sigma > 0:
                        deltas = [delta(s_val, pnl_strike, T_pnl, r, sigma, pnl_option_type) for s_val in greeks_spot_range]
                        gammas = [gamma(s_val, pnl_strike, T_pnl, r, sigma) for s_val in greeks_spot_range]
                        thetas = [theta(s_val, pnl_strike, T_pnl, r, sigma, pnl_option_type) for s_val in greeks_spot_range]
                        vegas = [vega(s_val, pnl_strike, T_pnl, r, sigma) for s_val in greeks_spot_range]

                        fig_greeks = go.Figure()
                        fig_greeks.add_trace(go.Scatter(x=greeks_spot_range, y=deltas, mode='lines', name='Delta'))
                        fig_greeks.add_trace(go.Scatter(x=greeks_spot_range, y=gammas, mode='lines', name='Gamma'))
                        fig_greeks.add_trace(go.Scatter(x=greeks_spot_range, y=thetas, mode='lines', name='Theta (Ann.)'))
                        fig_greeks.add_trace(go.Scatter(x=greeks_spot_range, y=vegas, mode='lines', name='Vega'))

                        fig_greeks.update_layout(
                            title=f"Option Greeks vs. Stock Price for Selected {pnl_option_type.upper()} Option",
                            xaxis_title="Stock Price",
                            yaxis_title="Greek Value",
                            hovermode="x unified",
                            height=600
                        )
                        st.plotly_chart(fig_greeks, use_container_width=True)
                    else:
                        st.warning("Cannot calculate Greeks plots: Time to Maturity or Volatility is zero.")

                else:
                    st.info("Please select an option to view P&L and Greeks plots.")
            else:
                st.info("No option chain data available for P&L and Greeks plots.")
        except Exception as e:
            st.error(f"Error generating P&L chart or Greeks plots: {e}")
    else:
        st.info("Please search for a stock on the 'Black-Scholes Calculator' tab to view P&L and Greeks Analysis.")

with tab4:
    st.header("Financial Terms Explained")
    st.markdown("Here you'll find explanations for key financial and analytical terms used in options trading and this application. Click on each term to expand its definition, relevant formulas, and importance.")

    with st.expander("Black-Scholes Model"):
        st.markdown(r"""
        The **Black-Scholes Model** is a mathematical model used to estimate the theoretical price of European-style options. It's a cornerstone in financial theory, providing a framework to understand how various factors influence option prices.
        
        **Formula (Call Option):**
        $C = S_0 N(d_1) - K e^{-rT} N(d_2)$
        
        **Formula (Put Option):**
        $P = K e^{-rT} N(-d_2) - S_0 N(-d_1)$
        
        Where:
        * $C$ = Call Option Price
        * $P$ = Put Option Price
        * $S_0$ = Current stock price (Spot Price)
        * $K$ = Strike price
        * $T$ = Time to maturity (in years)
        * $r$ = Risk-free interest rate
        * $\sigma$ = Volatility of the stock
        * $N(x)$ = Cumulative standard normal distribution function
        * $d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}$
        * $d_2 = d_1 - \sigma\sqrt{T}$
        
        **Importance:** It provides a theoretical fair value for options, helping traders identify potentially over- or undervalued options in the market. It's also the basis for calculating implied volatility and option Greeks.
        """)

    with st.expander("Option"):
        st.markdown("""
        An **option** is a financial derivative that gives the buyer the right, but not the obligation, to buy or sell an underlying asset (like a stock) at a predetermined price (the strike price) on or before a specific date (the expiration date).
        
        **Types:**
        * **Call Option:** Gives the holder the right to *buy* the underlying asset.
        * **Put Option:** Gives the holder the right to *sell* the underlying asset.
        
        **Importance:** Options are used for speculation (betting on price movements), hedging (reducing risk in other investments), and income generation.
        """)

    with st.expander("Call Option"):
        st.markdown("""
        A **Call Option** gives the buyer the right to **buy** an underlying asset at a specified **strike price** on or before the expiration date.
        
        **Profit Scenario (Long Call):** You profit if the underlying asset's price rises significantly above the strike price before expiration.
        
        **Importance:** Often bought when an investor is bullish (expects the price to rise) on an asset.
        """)
    
    with st.expander("Put Option"):
        st.markdown("""
        A **Put Option** gives the buyer the right to **sell** an underlying asset at a specified **strike price** on or before the expiration date.
        
        **Profit Scenario (Long Put):** You profit if the underlying asset's price falls significantly below the strike price before expiration.
        
        **Importance:** Often bought when an investor is bearish (expects the price to fall) on an asset, or used to hedge against potential losses in a long stock position.
        """)

    with st.expander("Strike Price (K)"):
        st.markdown("""
        The **Strike Price (K)** is the fixed price at which the owner of an option can buy (for a call) or sell (for a put) the underlying asset.
        
        **Importance:** It's a critical component of an option contract, determining whether an option is In-the-Money (ITM), At-the-Money (ATM), or Out-of-the-Money (OTM) relative to the current spot price, and directly impacts its value.
        """)

    with st.expander("Spot Price (S)"):
        st.markdown("""
        The **Spot Price (S)** is the current market price of the underlying asset (e.g., a stock) at any given moment.
        
        **Importance:** It's the most fundamental input for option pricing models, as the option's value is directly derived from its relationship to the current spot price and the strike price.
        """)

    with st.expander("Time to Maturity (T)"):
        st.markdown(r"""
        **Time to Maturity (T)**, also known as time to expiration, is the remaining time until an option contract expires. It is usually expressed in **years** in option pricing models.
        
        **Importance:** Time is a decaying asset for options. The longer the time to maturity, the more extrinsic value an option usually has (due to more time for the stock price to move), but this value erodes as time passes (Theta decay).
        """)

    with st.expander("Risk-Free Interest Rate (r)"):
        st.markdown("""
        The **Risk-Free Interest Rate (r)** is the theoretical rate of return of an investment with zero risk. In practice, this is often approximated by the yield on short-term government bonds (like US Treasury bills).
        
        **Importance:** It's an input in the Black-Scholes model because option pricing considers the time value of money, as well as the cost of financing an underlying position (for calls) or the return from holding cash (for puts).
        """)

    with st.expander("Volatility ($\sigma$)"):
        st.markdown(r"""
        **Volatility ($\sigma$)** measures the degree of variation of a trading price series over time. It's a key input in option pricing models, representing the expected fluctuations in the underlying asset's price.
        
        **Types:**
        * **Historical Volatility:** Calculated from past price movements of the underlying asset.
        * **Implied Volatility (IV):** Derived from the market price of an option. It's the volatility input that, when plugged into an option pricing model, yields the current market price of the option. It represents the market's *expectation* of future volatility.
        
        **Importance:** Higher volatility generally means higher option prices (for both calls and puts) because there's a greater chance for the stock price to make a significant move, increasing the probability of the option becoming profitable.
        """)

    with st.expander("Delta ($\Delta$)"):
        st.markdown(r"""
        **Delta ($\Delta$)** measures the **sensitivity of an option's price to a $1 change in the underlying asset's price.**
        
        **Formula (Call):** $\Delta_C = N(d_1)$
        **Formula (Put):** $\Delta_P = N(d_1) - 1$
        
        **Interpretation:**
        * A Call option's Delta is between 0 and 1 (e.g., 0.60 means the option price increases by $0.60 for every $1 stock price increase).
        * A Put option's Delta is between -1 and 0 (e.g., -0.40 means the option price decreases by $0.40 for every $1 stock price increase).
        
        **Importance:** It helps traders understand the directional exposure of their option position and can be used to hedge against changes in the underlying stock price.
        """)

    with st.expander("Gamma ($\Gamma$)"):
        st.markdown(r"""
        **Gamma ($\Gamma$)** measures the **rate of change of an option's Delta** with respect to a $1 change in the underlying asset's price. It's the second derivative of the option price with respect to the underlying price.
        
        **Formula:** $\Gamma = \frac{N'(d_1)}{S \sigma \sqrt{T}}$
        
        **Interpretation:** A high Gamma means that the option's Delta will change rapidly with small movements in the stock price. This can be beneficial for options buyers, as their Delta exposure can quickly increase, amplifying gains.
        
        **Importance:** Gamma is crucial for understanding how an option's directional exposure (Delta) will evolve. It's especially important for active traders who adjust their hedges.
        """)
    
    with st.expander("Theta ($\Theta$)"):
        st.markdown(r"""
        **Theta ($\Theta$)** measures the **sensitivity of an option's price to the passage of time.** It is also known as "time decay."
        
        **Formula (Call, per year):** $\Theta_C = -\frac{S N'(d_1) \sigma}{2\sqrt{T}} - r K e^{-rT} N(d_2)$
        **Formula (Put, per year):** $\Theta_P = -\frac{S N'(d_1) \sigma}{2\sqrt{T}} + r K e^{-rT} N(-d_2)$
        
        **Interpretation:** Theta is typically negative for long option positions, meaning the option loses value each day as it approaches expiration, all else being equal. A Theta of -0.05 (per day) means the option loses $0.05$ of its value daily.
        
        **Importance:** Theta highlights the cost of holding an option over time. Options sellers (short options) benefit from time decay, while options buyers (long options) pay for it.
        """)

    with st.expander("Vega ($\nu$)"):
        st.markdown(r"""
        **Vega ($\nu$)** measures the **sensitivity of an option's price to a 1% (0.01) change in the implied volatility** of the underlying asset.
        
        **Formula:** $\nu = S N'(d_1) \sqrt{T}$
        
        **Interpretation:** A positive Vega means that the option's price will increase if implied volatility rises, and decrease if implied volatility falls. For example, a Vega of 0.10 means the option price would increase by $0.10 for every 1% increase in implied volatility.
        
        **Importance:** Vega is critical for assessing volatility risk. Options buyers typically have positive Vega exposure, benefiting from increased market uncertainty, while options sellers have negative Vega.
        """)

    with st.expander("Rho ($\rho$)"):
        st.markdown(r"""
        **Rho ($\rho$)** measures the **sensitivity of an option's price to a 1% (0.01) change in the risk-free interest rate.**
        
        **Formula (Call):** $\rho_C = K T e^{-rT} N(d_2)$
        **Formula (Put):** $\rho_P = -K T e^{-rT} N(-d_2)$
        
        **Interpretation:**
        * For Call options, Rho is typically positive (higher interest rates mean higher call prices).
        * For Put options, Rho is typically negative (higher interest rates mean lower put prices).
        
        **Importance:** Rho is generally less significant than other Greeks unless interest rates are highly volatile or the option has a very long time to maturity.
        """)

    with st.expander("Volatility Smile/Skew"):
        st.markdown(r"""
        The **Volatility Smile** (or **Skew**) is a graphical pattern observed when plotting the implied volatility (IV) of options against their strike prices for a given expiration date.
        
        **Black-Scholes Assumption vs. Reality:** The Black-Scholes model assumes constant volatility across all strike prices, implying a flat line. However, in reality, out-of-the-money (OTM) and deep in-the-money (ITM) options often have higher implied volatilities than at-the-money (ATM) options, creating a "smile" or "smirk" shape.
        
        **Interpretation:**
        * **Smile (U-shape):** Often seen in currency options, suggesting the market expects large moves in either direction.
        * **Skew (downward slope):** Common in equity options, where lower strike (OTM Put) options have higher IVs, reflecting investor demand for downside protection.
        
        **Importance:** It highlights the limitations of the basic Black-Scholes model and provides insights into market sentiment regarding potential large price movements and perceived risk.
        """)

    with st.expander("In-the-Money (ITM)"):
        st.markdown("""
        An option is **In-the-Money (ITM)** if it has intrinsic value and would result in a profit if exercised immediately.
        
        * **Call Option:** Strike Price < Current Spot Price
        * **Put Option:** Strike Price > Current Spot Price
        
        **Importance:** ITM options are generally more expensive but have a higher probability of expiring profitably due to their existing intrinsic value.
        """)

    with st.expander("At-the-Money (ATM)"):
        st.markdown("""
        An option is **At-the-Money (ATM)** if its strike price is equal to or very close to the current spot price of the underlying asset.
        
        * **Call Option:** Strike Price $\approx$ Current Spot Price
        * **Put Option:** Strike Price $\approx$ Current Spot Price
        
        **Importance:** ATM options have the highest time value (extrinsic value) and are highly sensitive to changes in implied volatility.
        """)

    with st.expander("Out-of-the-Money (OTM)"):
        st.markdown("""
        An option is **Out-of-the-Money (OTM)** if it has no intrinsic value and would not result in a profit if exercised immediately.
        
        * **Call Option:** Strike Price > Current Spot Price
        * **Put Option:** Strike Price < Current Spot Price
        
        **Importance:** OTM options are generally cheaper and offer higher leverage (potential for large percentage gains) but have a lower probability of expiring profitably. Most OTM options expire worthless.
        """)

    with st.expander("P&L (Profit & Loss)"):
        st.markdown("""
        **P&L (Profit & Loss)** refers to the financial gain or loss from an investment or trade. In options trading, a P&L chart visualizes the potential profit or loss of an option strategy across a range of underlying asset prices at a specific point in time (usually expiration).
        
        **Calculation (Long Call at Expiry):** $P\&L = \text{max}(0, \text{Spot Price at Expiry} - \text{Strike Price}) - \text{Premium Paid}$
        
        **Calculation (Long Put at Expiry):** $P\&L = \text{max}(0, \text{Strike Price} - \text{Spot Price at Expiry}) - \text{Premium Paid}$
        
        **Importance:** P&L charts are essential for understanding the risk-reward profile of an option trade, identifying break-even points, maximum profit, and maximum loss.
        """)
