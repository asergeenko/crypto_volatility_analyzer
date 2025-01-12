import streamlit as st
from pycoingecko import CoinGeckoAPI
import pandas as pd
import numpy as np
from datetime import datetime
import time

NUM_TOP_COINS = 4

# Initialize CoinGecko API globally
cg = CoinGeckoAPI()

@st.cache_data(ttl=10800)
def get_top_coins(limit=NUM_TOP_COINS):
    try:
        coins = cg.get_coins_markets(
            vs_currency='usd',
            order='market_cap_desc',
            per_page=limit,
            sparkline=False
        )
        return [{'id': coin['id'], 'symbol': coin['symbol'].upper()} for coin in coins]
    except Exception as e:
        st.error(f"Error fetching top coins: {e}")
        return []

@st.cache_data(ttl=10800)
def get_historical_data(coin_id, days=30):
    try:
        ohlc = cg.get_coin_ohlc_by_id(
            id=coin_id,
            vs_currency='usd',
            days=days
        )

        df = pd.DataFrame(ohlc, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('date', inplace=True)
        return df

    except Exception as e:
        st.error(f"Error fetching data for {coin_id}: {e}")
        return None

def calculate_support_resistance_distances(data, months=3):
    """Calculate distances from current price to support and resistance levels using 3 month data"""
    if data is None:
        return None, None

    lookback_days = months * 30  # approximate days in N months

    # Take only last 3 months of data
    recent_data = data[-lookback_days:]

    if len(recent_data) < lookback_days / 2:
        return None, None

    # Calculate support as average of local minimums over 3 months
    lows = recent_data['low'].rolling(window=5).min()  # Find local minimums
    support_level = lows.mean()  # Average of all local minimums

    # Calculate resistance as average of local maximums over 3 months
    highs = recent_data['high'].rolling(window=5).max()  # Find local maximums
    resistance_level = highs.mean()  # Average of all local maximums

    current_price = data['close'].iloc[-1]

    # Calculate distances in percentages
    support_distance = ((current_price - support_level) / support_level * 100)
    resistance_distance = ((resistance_level - current_price) / current_price * 100)

    return support_distance, resistance_distance

def calculate_parkinson_volatility(data, window=30):
    """Calculate Parkinson volatility with median window"""
    if data is None or len(data) < window:
        return None

    log_hl = np.log(data['high'] / data['low'])**2
    daily_vol = np.sqrt(1 / (4 * np.log(2)) * log_hl)

    rolling_vol = daily_vol.rolling(window=window).median()
    volatility = rolling_vol.iloc[-1] * np.sqrt(252)

    return volatility * 100

@st.cache_data(ttl=10800)
def get_coin_id_mapping():
    try:
        coins_list = cg.get_coins_list()
        mapping = {}
        for coin in coins_list:
            symbol = coin['symbol'].lower()
            if symbol not in mapping:
                mapping[symbol] = coin['id']
        return mapping
    except Exception as e:
        st.error(f"Error fetching coins list: {e}")
        return {}

def get_coin_id(symbol, mapping):
    symbol = symbol.lower()
    return mapping.get(symbol)

@st.cache_data(ttl=10800)
def analyze_coins(coin_list=None):
    results = []

    if coin_list:
        mapping = get_coin_id_mapping()
        coins = []
        for symbol in coin_list:
            coin_id = get_coin_id(symbol, mapping)
            if coin_id:
                coins.append({'id': coin_id, 'symbol': symbol.upper()})
            else:
                st.warning(f"Could not find ID for symbol: {symbol}")
    else:
        coins = get_top_coins()

    total_coins = len(coins)
    if total_coins == 0:
        st.error("No valid coins to analyze!")
        return pd.DataFrame()

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, coin in enumerate(coins):
        progress = (i + 1) / total_coins
        progress_bar.progress(progress)
        status_text.text(f"Analyzing {coin['symbol']} ({i+1}/{total_coins})")

        data = get_historical_data(coin['id'])
        if data is not None:
            volatility = calculate_parkinson_volatility(data)
            support_dist, resistance_dist = calculate_support_resistance_distances(data)

            results.append({
                'symbol': coin['symbol'],
                'volatility': round(volatility, 2) if volatility else None,
                'support_dist': round(support_dist, 2) if support_dist else None,
                'resistance_dist': round(resistance_dist, 2) if resistance_dist else None
            })

        time.sleep(1.5)  # Respect API rate limits

    progress_bar.empty()
    status_text.empty()

    if not results:
        return pd.DataFrame(columns=['symbol', 'volatility', 'support_dist', 'resistance_dist'])

    return pd.DataFrame(results)

def apply_filters(df, min_vol, search):
    if df.empty:
        return df
    mask = df['volatility'] >= min_vol
    if search:
        mask &= df['symbol'].str.contains(search.upper())
    return df[mask]

def main():
    st.set_page_config(page_title="Crypto Volatility Analysis", layout="wide")

    st.title("Crypto Volatility Analysis")

    if 'data' not in st.session_state:
        st.session_state.data = None

    uploaded_file = st.file_uploader(
        f"Upload a text file with coin symbols (one per line) or leave empty for top {NUM_TOP_COINS} coins",
        type=['txt']
    )
    refresh = st.button("Refresh Data", type="primary")

    if refresh or (uploaded_file and st.session_state.data is None):
        st.cache_data.clear()
        st.session_state.data = None

    if st.session_state.data is None:
        coin_list = None
        if uploaded_file:
            content = uploaded_file.getvalue().decode()
            coin_list = [line.strip() for line in content.split('\n') if line.strip()]

        with st.spinner('Fetching and analyzing data...'):
            st.session_state.data = analyze_coins(coin_list)
            st.write("Data fetched:", len(st.session_state.data))

    # Filter controls
    col1, col2 = st.columns(2)
    with col1:
        min_vol = st.number_input("Minimum Volatility %", value=0.0)
    with col2:
        search = st.text_input("Search by Symbol")

    # Apply filters to cached data
    filtered_df = apply_filters(st.session_state.data, min_vol, search)
    st.write("Filtered data:", len(filtered_df))

    if not filtered_df.empty:
        # Rename columns for display
        display_df = filtered_df.copy()
        display_df.columns = [
            'Symbol',
            'Volatility %',
            'Support Distance %',
            'Resistance Distance %'
        ]

        # Display results
        st.dataframe(
            display_df.style
            .background_gradient(
                subset=['Volatility %'],
                cmap='Greens',  # Only green shades
                vmin=filtered_df['volatility'].min(),
                vmax=filtered_df['volatility'].max()
            )
            .format({
                'Volatility %': '{:,.2f}%',
                'Support Distance %': '{:,.2f}%',
                'Resistance Distance %': '{:,.2f}%'
            }),
            hide_index=True,
            use_container_width=True
        )

        # Add statistics
        st.subheader("Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Average Volatility",
                f"{filtered_df['volatility'].mean():.2f}%"
            )
        with col2:
            st.metric(
                "Median Volatility",
                f"{filtered_df['volatility'].median():.2f}%"
            )
        with col3:
            st.metric(
                "Coins Analyzed",
                len(filtered_df)
            )

        # Add timestamp
        st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    else:
        st.warning("No data to display. Try adjusting filters or uploading different symbols.")

if __name__ == "__main__":
    main()