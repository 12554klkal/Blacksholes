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
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def d2(S, K, T, r, sigma):
    """Calculates d2 for the Black-Scholes formula."""
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def bs_price(S, K, T, r, sigma, option_type="call"):
    """Black-Scholes formula to calculate option price."""
    # Ensure T > 0 for calculation, handle edge case if T is very small but positive
    if T <= 0:
        if option_type == "call":
            return max(0, S - K)
        else:
            return max(0, K - S)

    d1_val = d1(S, K, T, r, sigma)
    d2_val = d1_val - sigma * np.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1_val) - K * exp(-r * T) * norm.cdf(d2_val)
    else:
        return K * exp(-r * T) * norm.cdf(-d2_val) - S * norm.cdf(-d1_val)

def find_implied_volatility(market_price, S, K, T, r, option_type):
    """Finds implied volatility using Brent's method."""
    if T <= 0 or market_price <= 0:
        return np.nan
    try:
        # Define the function whose root we want to find
        # The function should return 0 when BS price equals market price
        func = lambda sigma: bs_price(S, K, T, r, sigma, option_type) - market_price
        
        # Use brentq to find the root within a reasonable range for volatility (0.0001 to 10)
        implied_vol = brentq(func, 0.0001, 10)
        return implied_vol
    except:
        # Return NaN if no root is found (e.g., market price is too far from theoretical bounds)
        return np.nan

# -------------------------
# Option Greeks Calculations
# -------------------------
def delta(S, K, T, r, sigma, option_type="call"):
    """Calculates Delta for Black-Scholes."""
    d1_val = d1(S, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(d1_val)
    else:
        return norm.cdf(d1_val) - 1

def gamma(S, K, T, r, sigma):
    """Calculates Gamma for Black-Scholes."""
    d1_val = d1(S, K, T, r, sigma)
    return norm.pdf(d1_val) / (S * sigma * np.sqrt(T))

def theta(S, K, T, r, sigma, option_type="call"):
    """Calculates Theta for Black-Scholes."""
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
    d1_val = d1(S, K, T, r, sigma)
    return S * norm.pdf(d1_val) * np.sqrt(T)

def rho(S, K, T, r, sigma, option_type="call"):
    """Calculates Rho for Black-Scholes."""
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
        returns = np.log(data['Adj Close'] / data['Adj Close'].shift(1)).dropna()
        volatility = returns.std() * np.sqrt(252) # Annualize by sqrt(252 trading days)
        return volatility
    except:
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
            S = ticker.info['regularMarketPrice']
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

        except (KeyError, IndexError):
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
    S_min = st.sidebar.number_input("Min Spot Price", value=default_s_min, step=1.0)
    S_max = st.sidebar.number_input("Max Spot Price", value=default_s_max, step=1.0)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Volatility Range for Heatmap")
    sigma_min = st.sidebar.slider("Min Volatility", 0.01, 1.0, 0.1, step=0.01)
    sigma_max = st.sidebar.slider("Max Volatility", 0.01, 1.0, 0.3, step=0.01)
    
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
                        font=dict(color="white" if prices[i,j] > mid_point else "black")
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
    # Options Screener (Moved to main tab for quick access)
    # -------------------------
    st.subheader("Highest Potential Profit Screener")
    if input_method == "Search for a Stock" and stock_symbol and S is not None:
        target_price = st.number_input(
            "Your Target Stock Price for Screener", 
            value=float(S), 
            step=1.0, 
            help="Enter the price you believe the stock will reach by expiry for this screener."
        )

        try:
            expiries = yf.Ticker(stock_symbol).options
            if not expiries:
                st.warning("No option chain data available for this stock for screener.")
            else:
                exp_date_screener = expiries[0]
                st.info(f"Analyzing options for expiry: **{exp_date_screener}**")

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
                            
                            if potential_profit > 0:
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
                            
                            if potential_profit > 0:
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
    st.markdown("Here you can analyze the sensitivity of the selected option to various market factors.")

    if S is not None and K is not None and T > 0 and r is not None and sigma is not None:
        st.subheader(f"Greeks for {selected_option_type.upper()} Option (S=${S:.2f}, K=${K:.2f}, T={T:.2f}yr, r={r:.2%}, σ={sigma:.2%})")
        
        col_delta, col_gamma, col_theta, col_vega, col_rho = st.columns(5)
        
        with col_delta:
            option_delta = delta(S, K, T, r, sigma, selected_option_type)
            st.metric("Delta", f"{option_delta:.3f}")
            st.write(f"Interpret: Expected change in option price for a $1 change in stock price. A {selected_option_type} option's price is expected to change by ${abs(option_delta):.2f}.")
        
        with col_gamma:
            option_gamma = gamma(S, K, T, r, sigma)
            st.metric("Gamma", f"{option_gamma:.3f}")
            st.write(f"Interpret: Measures the rate of change of Delta. High Gamma means Delta is very sensitive to stock price changes.")

        with col_theta:
            option_theta = theta(S, K, T, r, sigma, selected_option_type)
            st.metric("Theta (per year)", f"{option_theta:.3f}")
            st.write(f"Interpret: Time decay. The option price is expected to decrease by ${abs(option_theta / 365):.3f} per day.") # Convert to daily theta
        
        with col_vega:
            option_vega = vega(S, K, T, r, sigma)
            st.metric("Vega", f"{option_vega:.3f}")
            st.write(f"Interpret: Option's sensitivity to a 1% (0.01) change in volatility. A 1% rise in volatility would increase the option price by ${option_vega / 100:.2f}.")
            
        with col_rho:
            option_rho = rho(S, K, T, r, sigma, selected_option_type)
            st.metric("Rho", f"{option_rho:.3f}")
            st.write(f"Interpret: Option's sensitivity to a 1% (0.01) change in the risk-free rate. A 1% rise in interest rates would change the option price by ${option_rho / 100:.2f}.")
    else:
        st.info("Please ensure all Asset and Option Parameters are set correctly on the 'Black-Scholes Calculator' tab to view Greeks.")


with tab3:
    st.header("Volatility & P&L Analysis")
    st.markdown("Explore how implied volatility behaves across different strikes and visualize potential profit/loss scenarios.")

    if input_method == "Search for a Stock" and stock_symbol and S is not None:
        st.subheader("Volatility Smile/Skew")
        try:
            expiries = ticker.options
            if expiries:
                # Use the first expiry for volatility smile
                selected_smile_exp_date = st.selectbox("Select Expiration Date for Volatility Smile", expiries)

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
                        name='Call Options',
                        line=dict(color='blue')
                    ))
                    # Plot Puts
                    fig_smile.add_trace(go.Scatter(
                        x=df_smile[df_smile['Option Type'] == 'Put']['Strike'],
                        y=df_smile[df_smile['Option Type'] == 'Put']['Implied Volatility'],
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
                    st.info("Could not generate Volatility Smile. No valid option data with implied volatility found.")
            else:
                st.info("No option chain data available for volatility smile.")
        except Exception as e:
            st.error(f"Error generating Volatility Smile: {e}")
    else:
        st.info("Please search for a stock on the 'Black-Scholes Calculator' tab to view Volatility Smile.")

    st.markdown("---")
    st.subheader("Interactive P&L Chart & Greeks Plots")

    if input_method == "Search for a Stock" and stock_symbol and S is not None:
        try:
            expiries_pnl = ticker.options
            if expiries_pnl:
                selected_pnl_exp_date = st.selectbox("Select Expiration Date for P&L Chart", expiries_pnl)
                chain_pnl = ticker.option_chain(selected_pnl_exp_date)
                
                # Combine calls and puts for selection
                all_options_pnl = pd.concat([chain_pnl.calls, chain_pnl.puts])
                
                # Create display string for selectbox
                all_options_pnl['display_name'] = all_options_pnl.apply(
                    lambda row: f"{'Call' if row['contractSymbol'].endswith('C') else 'Put'} - Strike: {row['strike']:.2f} - Last Price: {row.get('lastPrice', row.get('bid')):.2f}",
                    axis=1
                )
                
                selected_option_pnl_str = st.selectbox(
                    "Select an Option to Analyze (P&L and Greeks Plots)", 
                    all_options_pnl['display_name'].tolist()
                )

                if selected_option_pnl_str:
                    # Extract the original row for the selected option
                    selected_option_row = all_options_pnl[all_options_pnl['display_name'] == selected_option_pnl_str].iloc[0]
                    
                    pnl_option_type = 'call' if selected_option_row['contractSymbol'].endswith('C') else 'put'
                    pnl_strike = selected_option_row['strike']
                    pnl_market_price = selected_option_row.get('lastPrice', selected_option_row.get('bid'))

                    days_to_expiry_pnl = (pd.to_datetime(selected_pnl_exp_date) - pd.Timestamp.now()).days
                    T_pnl = days_to_expiry_pnl / 365.0 if days_to_expiry_pnl > 0 else 0.01

                    # --- P&L Chart Calculation ---
                    spot_range = np.linspace(max(1, S - 50), S + 50, 100)
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
                    fig_pnl.add_hline(y=0, line_dash="dash", line_color="grey")
                    fig_pnl.add_vline(x=S, line_dash="dot", line_color="orange", annotation_text="Current Stock Price", annotation_position="top right")
                    fig_pnl.update_layout(
                        title=f"P&L at Expiry for Selected {pnl_option_type.upper()} Option",
                        xaxis_title="Stock Price at Expiration",
                        yaxis_title="Profit/Loss",
                        height=500
                    )
                    st.plotly_chart(fig_pnl, use_container_width=True)

                    st.markdown("---")
                    st.subheader("Greeks Plots vs. Stock Price")

                    # --- Greeks Plots Calculation ---
                    greeks_spot_range = np.linspace(max(1, S - 20), S + 20, 50) # Smaller range for Greeks sensitivity
                    deltas = [delta(s_val, pnl_strike, T_pnl, r, sigma, pnl_option_type) for s_val in greeks_spot_range]
                    gammas = [gamma(s_val, pnl_strike, T_pnl, r, sigma) for s_val in greeks_spot_range]
                    thetas = [theta(s_val, pnl_strike, T_pnl, r, sigma, pnl_option_type) for s_val in greeks_spot_range]
                    vegas = [vega(s_val, pnl_strike, T_pnl, r, sigma) for s_val in greeks_spot_range]

                    fig_greeks = go.Figure()
                    fig_greeks.add_trace(go.Scatter(x=greeks_spot_range, y=deltas, mode='lines', name='Delta'))
                    fig_greeks.add_trace(go.Scatter(x=greeks_spot_range, y=gammas, mode='lines', name='Gamma'))
                    fig_greeks.add_trace(go.Scatter(x=greeks_spot_range, y=thetas, mode='lines', name='Theta'))
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
                    st.info("Please select an option to view P&L and Greeks plots.")
            else:
                st.info("No option chain data available for P&L and Greeks plots.")
        except Exception as e:
            st.error(f"Error generating P&L chart or Greeks plots: {e}")
    else:
        st.info("Please search for a stock on the 'Black-Scholes Calculator' tab to view P&L and Greeks Analysis.")
