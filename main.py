import streamlit as st
from pycoingecko import CoinGeckoAPI
import pandas as pd
import numpy as np
from datetime import datetime
import time

NUM_TOP_COINS = 10

# Initialize CoinGecko API globally
cg = CoinGeckoAPI()

@st.cache_data(ttl=10800)  # Cache for 3 hours
def get_top_coins(limit=NUM_TOP_COINS):
    """Get top coins by market cap"""
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

@st.cache_data(ttl=10800)  # Cache for 3 hours
def get_historical_data(coin_id, days=30):
    """Get historical OHLC data"""
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

def calculate_standard_volatility(data, window=30):
    """Calculate standard volatility using close prices"""
    if data is None or len(data) < window:
        return None

    returns = np.log(data['close'] / data['close'].shift(1))
    rolling_std = returns.rolling(window=window).std()
    annualized_vol = rolling_std.iloc[-1] * np.sqrt(252) if not rolling_std.empty else None

    return annualized_vol * 100 if annualized_vol else None

def calculate_parkinson_volatility(data, window=30):
    """Calculate Parkinson volatility with median window"""
    if data is None or len(data) < window:
        return None

    log_hl = np.log(data['high'] / data['low'])**2
    daily_vol = np.sqrt(1 / (4 * np.log(2)) * log_hl)

    rolling_vol = daily_vol.rolling(window=window).median()
    annualized_vol = rolling_vol.iloc[-1] * np.sqrt(252) if not rolling_vol.empty else None

    return annualized_vol * 100 if annualized_vol else None

@st.cache_data(ttl=10800)
def get_coin_id_mapping():
    """Get mapping of coin symbols to their IDs"""
    try:
        # Get list of all coins
        coins_list = cg.get_coins_list()

        # Create mapping dictionary (symbol -> id)
        # Using lower case for case-insensitive matching
        mapping = {}
        for coin in coins_list:
            symbol = coin['symbol'].lower()
            # If we have multiple coins with same symbol, prefer the one with highest market cap
            if symbol not in mapping:
                mapping[symbol] = coin['id']
        return mapping
    except Exception as e:
        st.error(f"Error fetching coins list: {e}")
        return {}

def get_coin_id(symbol, mapping):
    """Get coin ID from symbol"""
    symbol = symbol.lower()
    return mapping.get(symbol)

@st.cache_data(ttl=10800)
def analyze_coins(coin_list=None):
    """Analyze specified coins or top coins if no list provided"""
    results = []

    # Get coins to analyze
    if coin_list:
        # Get mapping of symbols to IDs
        mapping = get_coin_id_mapping()

        # Convert symbols to IDs
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
            current_price = data['close'].iloc[-1]
            std_volatility = calculate_standard_volatility(data)
            park_volatility = calculate_parkinson_volatility(data)

            results.append({
                'symbol': coin['symbol'],
                'price': round(current_price, 4),
                'std_volatility': round(std_volatility, 2) if std_volatility else None,
                'park_volatility': round(park_volatility, 2) if park_volatility else None,
            })

        time.sleep(1.5)  # Respect API rate limits

    progress_bar.empty()
    status_text.empty()
    return pd.DataFrame(results)

def apply_filters(df, min_vol, max_vol, search, volatility_type='park_volatility'):
    """Apply filters to the dataframe without recomputing data"""
    mask = (df[volatility_type] >= min_vol) & (df[volatility_type] <= max_vol)
    if search:
        mask &= df['symbol'].str.contains(search.upper())
    return df[mask]

def main():
    st.set_page_config(page_title="Crypto Volatility Analysis", layout="wide")

    st.title("Crypto Volatility Analysis")
    st.markdown("""
    This dashboard shows volatility analysis for cryptocurrencies using both Standard and Parkinson methods.
    Data is cached for 3 hours to respect API limits.
    """)

    # Initialize session state for data
    if 'data' not in st.session_state:
        st.session_state.data = None


    # File upload and refresh section
    #col1, col2 = st.columns([3, 1])
    #with col1:
    uploaded_file = st.file_uploader(
        f"Upload a text file with coin symbols (one per line) or leave empty for top {NUM_TOP_COINS} coins",
        type=['txt']
    )
    #with col2:
        #st.write("")  # Add empty space to align with file uploader
    refresh = st.button("Refresh Data", type="primary")

    if refresh or (uploaded_file and st.session_state.data is None):
        st.cache_data.clear()
        st.session_state.data = None

    # Fetch or get cached data
    if st.session_state.data is None:
        coin_list = None
        if uploaded_file:
            content = uploaded_file.getvalue().decode()
            coin_list = [line.strip() for line in content.split('\n') if line.strip()]

        with st.spinner('Fetching and analyzing data...'):
            st.session_state.data = analyze_coins(coin_list)

    if not len(st.session_state.data):
        return

    # Filter controls
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        vol_type = st.selectbox(
            "Filter by Volatility Type",
            ['park_volatility', 'std_volatility'],
            format_func=lambda x: 'Parkinson' if x == 'park_volatility' else 'Standard Deviation'
        )
    with col2:
        min_vol = st.number_input("Minimum Volatility %", value=0.0)
    with col3:
        max_vol = st.number_input("Maximum Volatility %", value=1000.0)
    with col4:
        search = st.text_input("Search by Symbol")

    # Apply filters to cached data
    filtered_df = apply_filters(st.session_state.data, min_vol, max_vol, search, vol_type)

    # Rename columns
    display_df = filtered_df.copy()
    display_df.columns = [
        'Symbol',
        'Price USD',
        'Standard Volatility %',
        'Parkinson Volatility %'
    ]

    # Display results with both volatility columns colored
    st.dataframe(
        display_df.style
        .background_gradient(
            subset=['Standard Volatility %', 'Parkinson Volatility %'],
            cmap='RdYlGn_r'
        )
        .format({
            'Price USD': '${:,.2f}',
            'Standard Volatility %': '{:,.2f}%',
            'Parkinson Volatility %': '{:,.2f}%'
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
            f"{filtered_df[vol_type].mean():.2f}%"
        )
    with col2:
        st.metric(
            "Median Volatility",
            f"{filtered_df[vol_type].median():.2f}%"
        )
    with col3:
        st.metric(
            "Coins Analyzed",
            len(filtered_df)
        )

    # Add timestamp
    st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

if __name__ == "__main__":
    main()