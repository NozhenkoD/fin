import pandas as pd


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


def calculate_avg_volume(df: pd.DataFrame, window: int = 14) -> float:
    """
    Calculate average volume over a specified window.

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


def calculate_distance_from_sma(price: float, sma_value: float) -> float:
    """
    Calculate the ratio of current price to SMA (price/SMA).

    This indicates how far the price is from its moving average.
    Values > 1 mean price is above the SMA, < 1 mean below.

    Args:
        price: Current price
        sma_value: SMA value to compare against

    Returns:
        Ratio as float, or None if sma_value is invalid
    """
    if sma_value is None or sma_value == 0:
        return None

    return price / sma_value


def calculate_distance_from_ema(price: float, ema_value: float) -> float:
    """
    Calculate the ratio of current price to EMA (price/EMA).

    Args:
        price: Current price
        ema_value: EMA value to compare against

    Returns:
        Ratio as float, or None if ema_value is invalid
    """
    if ema_value is None or ema_value == 0:
        return None

    return price / ema_value


def get_latest_price(df: pd.DataFrame, price_col: str = 'Close') -> float:
    """
    Get the most recent price from the DataFrame.

    Args:
        df: DataFrame with OHLCV data
        price_col: Column name to use (default: 'Close')

    Returns:
        Latest price as float, or None if no data
    """
    if df.empty:
        return None

    return df[price_col].iloc[-1]


def get_latest_volume(df: pd.DataFrame) -> float:
    """
    Get the most recent volume from the DataFrame.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        Latest volume as float, or None if no data
    """
    if df.empty:
        return None

    return df['Volume'].iloc[-1]
