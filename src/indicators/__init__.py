"""
Technical Indicators Module

Provides wrappers around pandas_ta for calculating technical indicators.
"""

from .technical import (
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    calculate_bbands,
    calculate_atr,
    calculate_adx,
    add_indicators,
    # Legacy functions
    calculate_avg_volume,
    calculate_price_change
)

__all__ = [
    'calculate_sma',
    'calculate_ema',
    'calculate_rsi',
    'calculate_macd',
    'calculate_bbands',
    'calculate_atr',
    'calculate_adx',
    'add_indicators',
    'calculate_avg_volume',
    'calculate_price_change',
]
