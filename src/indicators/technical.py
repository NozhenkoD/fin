"""
Technical Indicators

Provides a clean interface to technical indicators with consistent API.
Uses pure pandas/numpy for robust, well-tested implementations.
"""

import pandas as pd
import numpy as np


# ============================================
# Moving Averages
# ============================================

def calculate_sma(df: pd.DataFrame, window: int, price_col: str = 'Close') -> pd.Series:
    """
    Calculate Simple Moving Average.

    Args:
        df: DataFrame with OHLCV data
        window: Number of periods for the moving average
        price_col: Column name to use for calculation (default: 'Close')

    Returns:
        Series containing SMA values
    """
    if df.empty or len(df) < window:
        return pd.Series(dtype=float)

    return df[price_col].rolling(window=window).mean()


def calculate_ema(df: pd.DataFrame, window: int, price_col: str = 'Close') -> pd.Series:
    """
    Calculate Exponential Moving Average.

    Args:
        df: DataFrame with OHLCV data
        window: Number of periods for the moving average
        price_col: Column name to use for calculation (default: 'Close')

    Returns:
        Series containing EMA values
    """
    if df.empty or len(df) < window:
        return pd.Series(dtype=float)

    return df[price_col].ewm(span=window, adjust=False).mean()


# ============================================
# Momentum Indicators
# ============================================

def calculate_rsi(df: pd.DataFrame, window: int = 14, price_col: str = 'Close') -> pd.Series:
    """
    Calculate Relative Strength Index (momentum oscillator).

    RSI ranges from 0 to 100:
    - Above 70: Overbought
    - Below 30: Oversold

    Args:
        df: DataFrame with OHLCV data
        window: Number of periods (default: 14)
        price_col: Column name to use (default: 'Close')

    Returns:
        Series containing RSI values
    """
    if df.empty or len(df) < window + 1:
        return pd.Series(dtype=float)

    # Calculate price changes
    delta = df[price_col].diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)

    # Calculate average gain and average loss using EMA
    avg_gain = gains.ewm(span=window, adjust=False).mean()
    avg_loss = losses.ewm(span=window, adjust=False).mean()

    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_macd(df: pd.DataFrame, price_col: str = 'Close',
                   fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    MACD shows the relationship between two moving averages.

    Args:
        df: DataFrame with OHLCV data
        price_col: Column name to use (default: 'Close')
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line period (default: 9)

    Returns:
        DataFrame with columns:
        - MACD_12_26_9: MACD line
        - MACDh_12_26_9: MACD histogram
        - MACDs_12_26_9: Signal line
    """
    if df.empty:
        return pd.DataFrame()

    # Calculate EMAs
    ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()

    # MACD line = fast EMA - slow EMA
    macd_line = ema_fast - ema_slow

    # Signal line = EMA of MACD line
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()

    # Histogram = MACD line - signal line
    histogram = macd_line - signal_line

    return pd.DataFrame({
        f'MACD_{fast}_{slow}_{signal}': macd_line,
        f'MACDh_{fast}_{slow}_{signal}': histogram,
        f'MACDs_{fast}_{slow}_{signal}': signal_line
    })


# ============================================
# Volatility Indicators
# ============================================

def calculate_bbands(df: pd.DataFrame, window: int = 20, std_dev: float = 2.0,
                     price_col: str = 'Close') -> pd.DataFrame:
    """
    Calculate Bollinger Bands (volatility indicator).

    Bollinger Bands consist of:
    - Middle band: SMA
    - Upper band: SMA + (std_dev * standard deviation)
    - Lower band: SMA - (std_dev * standard deviation)

    Args:
        df: DataFrame with OHLCV data
        window: Number of periods (default: 20)
        std_dev: Number of standard deviations (default: 2.0)
        price_col: Column name to use (default: 'Close')

    Returns:
        DataFrame with columns:
        - BBL_20_2.0: Lower band
        - BBM_20_2.0: Middle band (SMA)
        - BBU_20_2.0: Upper band
        - BBB_20_2.0: Bandwidth
        - BBP_20_2.0: %B (position within bands)
    """
    if df.empty or len(df) < window:
        return pd.DataFrame()

    # Calculate middle band (SMA)
    middle_band = df[price_col].rolling(window=window).mean()

    # Calculate standard deviation
    rolling_std = df[price_col].rolling(window=window).std()

    # Calculate upper and lower bands
    upper_band = middle_band + (rolling_std * std_dev)
    lower_band = middle_band - (rolling_std * std_dev)

    # Calculate bandwidth
    bandwidth = (upper_band - lower_band) / middle_band

    # Calculate %B (position within bands)
    percent_b = (df[price_col] - lower_band) / (upper_band - lower_band)

    return pd.DataFrame({
        f'BBL_{window}_{std_dev}': lower_band,
        f'BBM_{window}_{std_dev}': middle_band,
        f'BBU_{window}_{std_dev}': upper_band,
        f'BBB_{window}_{std_dev}': bandwidth,
        f'BBP_{window}_{std_dev}': percent_b
    })


def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculate Average True Range (volatility indicator).

    ATR measures market volatility by decomposing the entire range of price movement.

    Args:
        df: DataFrame with OHLCV data (must have High, Low, Close)
        window: Number of periods (default: 14)

    Returns:
        Series containing ATR values
    """
    if df.empty or len(df) < window:
        return pd.Series(dtype=float)

    # Calculate True Range
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # Calculate ATR as EMA of True Range
    atr = true_range.ewm(span=window, adjust=False).mean()

    return atr


# ============================================
# Trend Indicators
# ============================================

def calculate_adx(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculate Average Directional Index (trend strength indicator).

    ADX ranges from 0 to 100:
    - Below 20: Weak trend
    - 20-40: Strong trend
    - Above 40: Very strong trend

    Args:
        df: DataFrame with OHLCV data (must have High, Low, Close)
        window: Number of periods (default: 14)

    Returns:
        DataFrame with columns:
        - ADX_14: ADX value (trend strength)
        - DMP_14: +DI (positive directional indicator)
        - DMN_14: -DI (negative directional indicator)
    """
    if df.empty or len(df) < window:
        return pd.DataFrame()

    # Calculate directional movement
    high_diff = df['High'].diff()
    low_diff = -df['Low'].diff()

    # Calculate +DM and -DM
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)

    # Calculate True Range
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # Calculate smoothed values using EMA
    plus_dm_smooth = pd.Series(plus_dm, index=df.index).ewm(span=window, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm, index=df.index).ewm(span=window, adjust=False).mean()
    tr_smooth = true_range.ewm(span=window, adjust=False).mean()

    # Calculate +DI and -DI
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth

    # Calculate DX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)

    # Calculate ADX as EMA of DX
    adx = dx.ewm(span=window, adjust=False).mean()

    return pd.DataFrame({
        f'ADX_{window}': adx,
        f'DMP_{window}': plus_di,
        f'DMN_{window}': minus_di
    })


# ============================================
# Batch Helper Function
# ============================================

def add_indicators(df: pd.DataFrame, indicators: list) -> pd.DataFrame:
    """
    Add multiple indicators to a DataFrame in one call.

    Args:
        df: DataFrame with OHLCV data
        indicators: List of indicator specs, e.g.:
            ['SMA50', 'SMA200', 'RSI14', 'MACD', 'BBANDS20', 'ATR14']

    Returns:
        DataFrame with indicator columns added

    Example:
        df = add_indicators(df, ['SMA200', 'RSI14', 'MACD'])

    Supported indicators:
        - SMA{window}: Simple Moving Average (e.g., SMA50, SMA200)
        - EMA{window}: Exponential Moving Average (e.g., EMA20, EMA50)
        - RSI{window}: Relative Strength Index (e.g., RSI14)
        - MACD: MACD indicator
        - BBANDS{window}: Bollinger Bands (e.g., BBANDS20)
        - ATR{window}: Average True Range (e.g., ATR14)
        - ADX{window}: Average Directional Index (e.g., ADX14)
    """
    df = df.copy()

    for indicator in indicators:
        indicator_upper = indicator.upper()

        # SMA with window
        if indicator_upper.startswith('SMA'):
            window = int(indicator[3:])
            sma = calculate_sma(df, window)
            df[f'SMA{window}'] = sma

        # EMA with window
        elif indicator_upper.startswith('EMA'):
            window = int(indicator[3:])
            ema = calculate_ema(df, window)
            df[f'EMA{window}'] = ema

        # RSI with window (default 14)
        elif indicator_upper.startswith('RSI'):
            window = int(indicator[3:]) if len(indicator) > 3 else 14
            rsi = calculate_rsi(df, window)
            df[f'RSI{window}'] = rsi

        # MACD (returns 3 columns)
        elif indicator_upper == 'MACD':
            macd_df = calculate_macd(df)
            if not macd_df.empty:
                df = pd.concat([df, macd_df], axis=1)

        # Bollinger Bands (returns 5 columns)
        elif indicator_upper.startswith('BBANDS'):
            window = int(indicator[6:]) if len(indicator) > 6 else 20
            bbands_df = calculate_bbands(df, window)
            if not bbands_df.empty:
                df = pd.concat([df, bbands_df], axis=1)

        # ATR with window
        elif indicator_upper.startswith('ATR'):
            window = int(indicator[3:]) if len(indicator) > 3 else 14
            atr = calculate_atr(df, window)
            df[f'ATR{window}'] = atr

        # ADX with window
        elif indicator_upper.startswith('ADX'):
            window = int(indicator[3:]) if len(indicator) > 3 else 14
            adx_df = calculate_adx(df, window)
            if not adx_df.empty:
                df = pd.concat([df, adx_df], axis=1)

    return df


# ============================================
# Legacy Functions (Backward Compatibility)
# ============================================

def calculate_avg_volume(df: pd.DataFrame, window: int = 14) -> float:
    """
    Calculate average volume over a specified window.

    This is a legacy function from calculator.py for backward compatibility.

    Args:
        df: DataFrame with OHLCV data
        window: Number of periods to average (default: 14)

    Returns:
        Average volume as float, or None if insufficient data
    """
    if df.empty or len(df) < window:
        return None

    return df['Volume'].tail(window).mean()


def calculate_price_change(df: pd.DataFrame) -> float:
    """
    Calculate percentage change from previous close to current close.

    This is a legacy function from calculator.py for backward compatibility.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        Percentage change as float, or None if insufficient data
    """
    if df.empty or len(df) < 2:
        return None

    current_price = df['Close'].iloc[-1]
    previous_price = df['Close'].iloc[-2]

    if previous_price == 0:
        return None

    change = ((current_price - previous_price) / previous_price) * 100
    return change
