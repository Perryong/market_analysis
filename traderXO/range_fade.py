"""
Range Fading Strategy Plotting
Focus on identifying and trading range boundaries with reversal signals
"""

# Set matplotlib backend before imports
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, FancyBboxPatch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from traderXO.data_manager import DataManager
from traderXO.indicators import TechnicalIndicators
from traderXO.key_levels import KeyLevels


def plot_range_fade(ticker: str, range_high: float = None, range_low: float = None,
                    timeframe: str = '4h', lookback_days: int = 60,
                    exchange: str = 'binance', save_dir: str = 'traderXO/plots'):
    """
    Plot Range Fading Strategy
    
    Identify range boundaries and fade extremes with reversal signals
    
    Args:
        ticker: Trading pair (e.g., 'BTC/USDT')
        range_high: Manual range high (auto-detected if None)
        range_low: Manual range low (auto-detected if None)
        timeframe: Chart timeframe
        lookback_days: Days of historical data
        exchange: Exchange name
        save_dir: Directory to save plots
    """
    print(f"\n{'='*70}")
    print(f"RANGE FADING STRATEGY - {ticker}")
    print(f"{'='*70}\n")
    
    # Initialize data manager
    dm = DataManager(exchange)
    
    # Fetch data
    df = dm.fetch_ohlcv(ticker, timeframe, lookback_days)
    weekly_df = dm.fetch_ohlcv(ticker, '1w', lookback_days * 2)
    
    # Calculate indicators
    df = TechnicalIndicators.add_all_indicators(df, weekly_df)
    
    # Auto-detect range if not provided
    if range_high is None or range_low is None:
        lookback_for_range = 40  # Use last 40 candles for range
        recent_df = df.tail(lookback_for_range)
        range_high = recent_df['high'].max()
        range_low = recent_df['low'].min()
        print(f"[AUTO] Detected range: ${range_low:.2f} - ${range_high:.2f}")
    
    # Calculate range metrics
    range_mid = (range_high + range_low) / 2
    range_size = range_high - range_low
    
    # Define trade zones
    short_zone_high = range_high
    short_zone_low = range_high - (range_size * 0.15)  # Top 15%
    
    long_zone_high = range_low + (range_size * 0.15)  # Bottom 15%
    long_zone_low = range_low
    
    neutral_zone_high = range_mid + (range_size * 0.15)
    neutral_zone_low = range_mid - (range_size * 0.15)
    
    # Create figure with 2 subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.05)
    
    ax1 = fig.add_subplot(gs[0])  # Main price chart
    ax2 = fig.add_subplot(gs[1], sharex=ax1)  # Reversal signals
    
    # ===== MAIN CHART =====
    plot_range_main_chart(ax1, df, ticker, timeframe, 
                         range_high, range_low, range_mid,
                         short_zone_high, short_zone_low,
                         long_zone_high, long_zone_low,
                         neutral_zone_high, neutral_zone_low)
    
    # ===== REVERSAL SIGNALS PANEL =====
    plot_reversal_signals(ax2, df, range_high, range_low)
    
    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Hide x-axis labels for upper plot
    plt.setp(ax1.get_xticklabels(), visible=False)
    
    # Save plot
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime('%Y-%m-%d')
    filename = f"{ticker.replace('/', '_')}_{timeframe}_range_fade_{date_str}.png"
    filepath = Path(save_dir) / filename
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[SUCCESS] Range Fade plot saved: {filepath}")
    return filepath


def plot_range_main_chart(ax, df, ticker, timeframe, range_high, range_low, range_mid,
                         short_zone_high, short_zone_low, long_zone_high, long_zone_low,
                         neutral_zone_high, neutral_zone_low):
    """Plot main range chart with zones"""
    
    # Plot candlesticks
    for idx, row in df.iterrows():
        color = '#26a69a' if row['close'] >= row['open'] else '#ef5350'
        
        # Candle body
        height = abs(row['close'] - row['open'])
        bottom = min(row['open'], row['close'])
        rect = Rectangle((mdates.date2num(idx), bottom), 0.002, height,
                        facecolor=color, edgecolor=color, alpha=0.9)
        ax.add_patch(rect)
        
        # Wicks
        ax.plot([mdates.date2num(idx), mdates.date2num(idx)],
               [row['low'], row['high']], color=color, linewidth=0.8, alpha=0.9)
    
    # Plot 12/21 Weekly EMAs
    if 'ema_12_weekly' in df.columns:
        ax.plot(df.index, df['ema_12_weekly'], label='12 EMA (Weekly)', 
               color='blue', linewidth=2, alpha=0.7)
    if 'ema_21_weekly' in df.columns:
        ax.plot(df.index, df['ema_21_weekly'], label='21 EMA (Weekly)', 
               color='purple', linewidth=2, alpha=0.7)
    
    # Range boundaries (thick lines)
    ax.axhline(range_high, color='red', linestyle='-', linewidth=4, 
              alpha=0.8, label='Range High', zorder=3)
    ax.axhline(range_low, color='green', linestyle='-', linewidth=4, 
              alpha=0.8, label='Range Low', zorder=3)
    
    # Range midpoint (dashed line)
    ax.axhline(range_mid, color='gray', linestyle='--', linewidth=2, 
              alpha=0.6, label='Range Mid', zorder=3)
    
    # Short zone (red box near range high)
    width = mdates.date2num(df.index[-1]) - mdates.date2num(df.index[0])
    short_zone = FancyBboxPatch(
        (mdates.date2num(df.index[0]), short_zone_low),
        width, short_zone_high - short_zone_low,
        boxstyle="round,pad=0.01",
        edgecolor='red', facecolor='red',
        alpha=0.15, linewidth=2, zorder=1
    )
    ax.add_patch(short_zone)
    ax.text(mdates.date2num(df.index[len(df)//2]), short_zone_high - 10,
           'SHORT ZONE', fontsize=12, color='red', weight='bold',
           horizontalalignment='center', verticalalignment='top')
    
    # Long zone (green box near range low)
    long_zone = FancyBboxPatch(
        (mdates.date2num(df.index[0]), long_zone_low),
        width, long_zone_high - long_zone_low,
        boxstyle="round,pad=0.01",
        edgecolor='green', facecolor='green',
        alpha=0.15, linewidth=2, zorder=1
    )
    ax.add_patch(long_zone)
    ax.text(mdates.date2num(df.index[len(df)//2]), long_zone_low + 10,
           'LONG ZONE', fontsize=12, color='green', weight='bold',
           horizontalalignment='center', verticalalignment='bottom')
    
    # Neutral zone (grey in middle)
    neutral_zone = FancyBboxPatch(
        (mdates.date2num(df.index[0]), neutral_zone_low),
        width, neutral_zone_high - neutral_zone_low,
        boxstyle="round,pad=0.01",
        edgecolor='gray', facecolor='gray',
        alpha=0.08, linewidth=1, linestyle='--', zorder=0
    )
    ax.add_patch(neutral_zone)
    ax.text(mdates.date2num(df.index[len(df)//2]), range_mid,
           'NEUTRAL ZONE', fontsize=11, color='gray', weight='bold',
           horizontalalignment='center', verticalalignment='center', alpha=0.6)
    
    # Entry/Exit signals based on order flow logic
    # Long entries: Price in long zone + RSI oversold + positive delta
    if 'rsi' in df.columns and 'delta' in df.columns:
        long_entry_mask = (
            (df['low'] <= long_zone_high) &
            (df['rsi'] < 35) &
            (df['delta'] > 0) &
            (df['close'] > df['open'])
        )
        ax.scatter(df[long_entry_mask].index, df[long_entry_mask]['low'],
                  marker='^', s=200, color='lime', edgecolor='darkgreen', linewidth=3,
                  label='LONG Entry', zorder=5)
        
        # Short entries: Price in short zone + RSI overbought + negative delta
        short_entry_mask = (
            (df['high'] >= short_zone_low) &
            (df['rsi'] > 65) &
            (df['delta'] < 0) &
            (df['close'] < df['open'])
        )
        ax.scatter(df[short_entry_mask].index, df[short_entry_mask]['high'],
                  marker='v', s=200, color='red', edgecolor='darkred', linewidth=3,
                  label='SHORT Entry', zorder=5)
        
        # Exit signals: Price reaching opposite zone
        long_exit_mask = (df['high'] >= range_mid) & (df.index > df[long_entry_mask].index[0] if long_entry_mask.any() else False)
        short_exit_mask = (df['low'] <= range_mid) & (df.index > df[short_entry_mask].index[0] if short_entry_mask.any() else False)
        
        ax.scatter(df[long_exit_mask].index, df[long_exit_mask]['high'],
                  marker='x', s=150, color='orange', linewidth=3,
                  label='Exit Signal', zorder=5)
        ax.scatter(df[short_exit_mask].index, df[short_exit_mask]['low'],
                  marker='x', s=150, color='orange', linewidth=3, zorder=5)
    
    # Origins/pivot points (swing highs/lows)
    swing_high_mask = (
        (df['high'] > df['high'].shift(1)) &
        (df['high'] > df['high'].shift(-1)) &
        (df['high'] >= range_mid)
    )
    swing_low_mask = (
        (df['low'] < df['low'].shift(1)) &
        (df['low'] < df['low'].shift(-1)) &
        (df['low'] <= range_mid)
    )
    
    ax.scatter(df[swing_high_mask].index, df[swing_high_mask]['high'],
              marker='D', s=80, color='purple', edgecolor='white', linewidth=1,
              label='Pivot High', zorder=4, alpha=0.7)
    ax.scatter(df[swing_low_mask].index, df[swing_low_mask]['low'],
              marker='D', s=80, color='orange', edgecolor='white', linewidth=1,
              label='Pivot Low', zorder=4, alpha=0.7)
    
    # Current price and range metrics
    current_price = df['close'].iloc[-1]
    range_position = ((current_price - range_low) / (range_high - range_low)) * 100
    
    info_text = f'''Current: ${current_price:.2f}
Range: ${range_low:.2f} - ${range_high:.2f}
Position: {range_position:.1f}% of range
Mid: ${range_mid:.2f}'''
    
    ax.text(0.02, 0.98, info_text, 
           transform=ax.transAxes, fontsize=11, weight='bold',
           verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Risk/Reward for current position
    if current_price <= long_zone_high:
        risk = current_price - range_low
        reward = range_mid - current_price
        rr_ratio = reward / risk if risk > 0 else 0
        
        rr_text = f'''LONG Setup
Entry: ${current_price:.2f}
Stop: ${range_low:.2f}
Target: ${range_mid:.2f}
R/R: {rr_ratio:.2f}'''
        
        ax.text(0.98, 0.02, rr_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
               
    elif current_price >= short_zone_low:
        risk = range_high - current_price
        reward = current_price - range_mid
        rr_ratio = reward / risk if risk > 0 else 0
        
        rr_text = f'''SHORT Setup
Entry: ${current_price:.2f}
Stop: ${range_high:.2f}
Target: ${range_mid:.2f}
R/R: {rr_ratio:.2f}'''
        
        ax.text(0.98, 0.02, rr_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    ax.set_ylabel('Price', fontsize=12, weight='bold')
    ax.set_title(f'{ticker} - Range Fading Strategy ({timeframe})', 
                fontsize=14, weight='bold', pad=10)
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)


def plot_reversal_signals(ax, df, range_high, range_low):
    """Plot reversal signals panel with RSI and volume"""
    
    # RSI
    if 'rsi' in df.columns:
        ax.plot(df.index, df['rsi'], color='purple', linewidth=2, label='RSI')
        ax.axhline(70, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Overbought')
        ax.axhline(30, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Oversold')
        ax.axhline(50, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        
        # Highlight exhaustion zones
        exhaustion_high = (df['rsi'] > 70) & (df['close'] >= range_high * 0.95)
        exhaustion_low = (df['rsi'] < 30) & (df['close'] <= range_low * 1.05)
        
        ax.fill_between(df[exhaustion_high].index, 0, 100, 
                       color='red', alpha=0.2, label='Exhaustion High')
        ax.fill_between(df[exhaustion_low].index, 0, 100, 
                       color='green', alpha=0.2, label='Exhaustion Low')
    
    # Volume spikes (secondary axis)
    ax2 = ax.twinx()
    colors = ['#26a69a' if row['close'] >= row['open'] else '#ef5350' 
              for _, row in df.iterrows()]
    ax2.bar(df.index, df['volume'], color=colors, alpha=0.3, width=0.002, label='Volume')
    
    # Highlight volume spikes at extremes
    at_high = df['close'] >= range_high * 0.98
    at_low = df['close'] <= range_low * 1.02
    vol_threshold = df['volume'].mean() * 2
    
    spike_high_mask = at_high & (df['volume'] > vol_threshold)
    spike_low_mask = at_low & (df['volume'] > vol_threshold)
    
    ax2.bar(df[spike_high_mask].index, df[spike_high_mask]['volume'],
           color='yellow', alpha=0.8, width=0.002, edgecolor='red', linewidth=2)
    ax2.bar(df[spike_low_mask].index, df[spike_low_mask]['volume'],
           color='yellow', alpha=0.8, width=0.002, edgecolor='green', linewidth=2)
    
    ax.set_ylabel('RSI', fontsize=10, weight='bold')
    ax2.set_ylabel('Volume', fontsize=10)
    ax.set_xlabel('Date/Time', fontsize=10, weight='bold')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

