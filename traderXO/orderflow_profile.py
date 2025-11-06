"""
Order Flow & Market Profile Plotting Strategy
Displays price action, order flow, delta, and market structure
"""

# Set matplotlib backend before imports
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional

from traderXO.data_manager import DataManager
from traderXO.indicators import TechnicalIndicators
from traderXO.key_levels import KeyLevels


def plot_orderflow_profile(ticker: str, timeframe: str = '1h', lookback_days: int = 30,
                           exchange: str = 'binance', save_dir: str = 'traderXO/plots'):
    """
    Plot Order Flow & Market Profile strategy
    
    Main Chart: Candlesticks with EMAs, VWAP, key levels
    Middle Panel: Delta histogram and cumulative delta
    Lower Panel: ATR and volume
    
    Args:
        ticker: Trading pair (e.g., 'BTC/USDT')
        timeframe: Chart timeframe
        lookback_days: Days of historical data
        exchange: Exchange name
        save_dir: Directory to save plots
    """
    print(f"\n{'='*70}")
    print(f"ORDER FLOW & MARKET PROFILE STRATEGY - {ticker}")
    print(f"{'='*70}\n")
    
    # Initialize data manager
    dm = DataManager(exchange)
    
    # Fetch data
    df = dm.fetch_ohlcv(ticker, timeframe, lookback_days)
    weekly_df = dm.fetch_ohlcv(ticker, '1w', lookback_days * 2)
    
    # Calculate indicators
    df = TechnicalIndicators.add_all_indicators(df, weekly_df)
    
    # Calculate key levels
    df['monthly_open'] = KeyLevels.monthly_opens(df)
    df['quarterly_open'] = KeyLevels.quarterly_opens(df)
    df['onh'], df['onl'] = KeyLevels.session_levels(df)
    
    market_profile = KeyLevels.market_profile(df)
    df['poc'] = market_profile['poc']
    df['vah'] = market_profile['vah']
    df['val'] = market_profile['val']
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.05)
    
    ax1 = fig.add_subplot(gs[0])  # Main price chart
    ax2 = fig.add_subplot(gs[1], sharex=ax1)  # Delta
    ax3 = fig.add_subplot(gs[2], sharex=ax1)  # ATR & Volume
    
    # ===== MAIN CHART =====
    plot_main_chart(ax1, df, ticker, timeframe)
    
    # ===== DELTA PANEL =====
    plot_delta_panel(ax2, df)
    
    # ===== ATR & VOLUME PANEL =====
    plot_atr_volume_panel(ax3, df)
    
    # Format x-axis
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Hide x-axis labels for upper plots
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    
    # Save plot
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime('%Y-%m-%d')
    filename = f"{ticker.replace('/', '_')}_{timeframe}_orderflow_{date_str}.png"
    filepath = Path(save_dir) / filename
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[SUCCESS] Order Flow plot saved: {filepath}")
    return filepath


def plot_main_chart(ax, df, ticker, timeframe):
    """Plot main price chart with EMAs, VWAP, and key levels"""
    
    # Color zones based on EMA position
    if 'ema_12_weekly' in df.columns:
        # Bullish zone (above 12 EMA)
        bullish_mask = df['close'] > df['ema_12_weekly']
        for idx in df[bullish_mask].index:
            ax.axvspan(idx, idx + pd.Timedelta(hours=1), alpha=0.05, color='green', zorder=0)
        
        # Bearish zone (below 21 EMA)
        if 'ema_21_weekly' in df.columns:
            bearish_mask = df['close'] < df['ema_21_weekly']
            for idx in df[bearish_mask].index:
                ax.axvspan(idx, idx + pd.Timedelta(hours=1), alpha=0.05, color='red', zorder=0)
    
    # Plot candlesticks
    for idx, row in df.iterrows():
        color = 'green' if row['close'] >= row['open'] else 'red'
        
        # Candle body
        height = abs(row['close'] - row['open'])
        bottom = min(row['open'], row['close'])
        ax.add_patch(Rectangle((mdates.date2num(idx), bottom), 0.0003, height,
                               facecolor=color, edgecolor=color, alpha=0.8))
        
        # Wicks
        ax.plot([mdates.date2num(idx), mdates.date2num(idx)],
               [row['low'], row['high']], color=color, linewidth=0.5, alpha=0.8)
    
    # Plot EMAs
    if 'ema_12_weekly' in df.columns:
        ax.plot(df.index, df['ema_12_weekly'], label='12 EMA (Weekly)', 
               color='blue', linewidth=2, alpha=0.8)
    if 'ema_21_weekly' in df.columns:
        ax.plot(df.index, df['ema_21_weekly'], label='21 EMA (Weekly)', 
               color='purple', linewidth=2, alpha=0.8)
    
    ax.plot(df.index, df['ema_20'], label='20 EMA', color='orange', 
           linewidth=1.5, alpha=0.7, linestyle='--')
    ax.plot(df.index, df['ema_50'], label='50 EMA', color='brown', 
           linewidth=1.5, alpha=0.7, linestyle='--')
    
    # VWAP
    ax.plot(df.index, df['vwap'], label='VWAP', color='cyan', 
           linewidth=2, alpha=0.8, linestyle='-.')
    
    # Key horizontal levels
    current_monthly = df['monthly_open'].iloc[-1]
    ax.axhline(current_monthly, color='darkblue', linestyle='-', linewidth=2, 
              alpha=0.6, label='Monthly Open')
    
    current_quarterly = df['quarterly_open'].iloc[-1]
    ax.axhline(current_quarterly, color='darkgreen', linestyle='--', linewidth=2, 
              alpha=0.6, label='Quarterly Open')
    
    # Session levels (ONH/ONL)
    current_onh = df['onh'].iloc[-1]
    current_onl = df['onl'].iloc[-1]
    ax.axhline(current_onh, color='red', linestyle=':', linewidth=1.5, 
              alpha=0.5, label='ONH')
    ax.axhline(current_onl, color='green', linestyle=':', linewidth=1.5, 
              alpha=0.5, label='ONL')
    
    # Market Profile levels
    current_poc = df['poc'].iloc[-1]
    current_vah = df['vah'].iloc[-1]
    current_val = df['val'].iloc[-1]
    
    ax.axhline(current_poc, color='yellow', linestyle='-', linewidth=2, 
              alpha=0.7, label='POC')
    ax.axhline(current_vah, color='orange', linestyle='--', linewidth=1, 
              alpha=0.5, label='VAH')
    ax.axhline(current_val, color='orange', linestyle='--', linewidth=1, 
              alpha=0.5, label='VAL')
    
    # Annotations
    current_price = df['close'].iloc[-1]
    ax.text(0.02, 0.98, f'Current: ${current_price:.2f}', 
           transform=ax.transAxes, fontsize=12, weight='bold',
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_ylabel('Price', fontsize=12, weight='bold')
    ax.set_title(f'{ticker} - Order Flow & Market Profile ({timeframe})', 
                fontsize=14, weight='bold', pad=10)
    ax.legend(loc='upper left', fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)


def plot_delta_panel(ax, df):
    """Plot delta histogram and cumulative delta"""
    
    # Delta histogram
    colors = ['green' if x > 0 else 'red' for x in df['delta']]
    ax.bar(df.index, df['delta'], color=colors, alpha=0.6, width=0.0003, label='Delta')
    
    # Cumulative delta line
    ax2 = ax.twinx()
    ax2.plot(df.index, df['cumulative_delta'], color='blue', linewidth=2, 
            alpha=0.8, label='Cumulative Delta')
    ax2.set_ylabel('Cumulative Delta', fontsize=10)
    
    # Highlight absorption zones (high delta, low price movement)
    price_change_pct = ((df['close'] - df['open']) / df['open'] * 100).abs()
    delta_normalized = df['delta'].abs() / df['volume']
    absorption_mask = (delta_normalized > 0.3) & (price_change_pct < 0.5)
    
    for idx in df[absorption_mask].index:
        ax.axvspan(idx, idx + pd.Timedelta(hours=1), alpha=0.15, color='yellow', zorder=0)
    
    ax.set_ylabel('Delta', fontsize=10, weight='bold')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_atr_volume_panel(ax, df):
    """Plot ATR and volume with market structure"""
    
    # Volume bars
    colors = ['green' if row['close'] >= row['open'] else 'red' 
              for _, row in df.iterrows()]
    
    ax2 = ax.twinx()
    ax2.bar(df.index, df['volume'], color=colors, alpha=0.4, width=0.0003, label='Volume')
    
    # Highlight high volume (yellow/red)
    high_volume_threshold = df['volume'].mean() * 2
    high_vol_mask = df['volume'] > high_volume_threshold
    ax2.bar(df[high_vol_mask].index, df[high_vol_mask]['volume'], 
           color='yellow', alpha=0.7, width=0.0003, edgecolor='red', linewidth=1)
    
    # ATR
    ax.plot(df.index, df['atr'], color='purple', linewidth=2, label='ATR')
    ax.fill_between(df.index, 0, df['atr'], alpha=0.2, color='purple')
    
    # ATR expansion/compression zones
    atr_ma = df['atr'].rolling(window=20).mean()
    expansion_mask = df['atr'] > atr_ma * 1.5
    compression_mask = df['atr'] < atr_ma * 0.7
    
    ax.scatter(df[expansion_mask].index, df[expansion_mask]['atr'], 
              color='red', marker='^', s=50, label='ATR Expansion', zorder=5)
    ax.scatter(df[compression_mask].index, df[compression_mask]['atr'], 
              color='blue', marker='v', s=50, label='ATR Compression', zorder=5)
    
    ax.set_ylabel('ATR', fontsize=10, weight='bold')
    ax2.set_ylabel('Volume', fontsize=10)
    ax.set_xlabel('Date/Time', fontsize=10, weight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

