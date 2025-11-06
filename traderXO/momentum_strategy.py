"""
Momentum Strategy Plotting
Focus on weekly/daily 12/21 EMAs with momentum and volume analysis
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
from typing import Optional

from traderXO.data_manager import DataManager
from traderXO.indicators import TechnicalIndicators
from traderXO.key_levels import KeyLevels


def plot_momentum_strategy(ticker: str, timeframe: str = '1d', lookback_days: int = 180,
                           exchange: str = 'binance', benchmark_ticker: Optional[str] = None,
                           save_dir: str = 'traderXO/plots'):
    """
    Plot Momentum Strategy
    
    Focus on 12/21 Weekly EMAs, monthly opens, demand zones, and breakouts
    
    Args:
        ticker: Trading pair (e.g., 'BTC/USDT')
        timeframe: Chart timeframe ('1d' or '1w')
        lookback_days: Days of historical data
        exchange: Exchange name
        benchmark_ticker: Optional benchmark for relative strength
        save_dir: Directory to save plots
    """
    print(f"\n{'='*70}")
    print(f"MOMENTUM STRATEGY - {ticker}")
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
    
    # Identify demand/supply zones
    demand_zones = KeyLevels.identify_demand_zones(df)
    supply_zones = KeyLevels.identify_supply_zones(df)
    
    # Relative strength if benchmark provided
    rs_data = None
    if benchmark_ticker:
        try:
            benchmark_df = dm.fetch_ohlcv(benchmark_ticker, timeframe, lookback_days)
            rs_data = TechnicalIndicators.relative_strength(df, benchmark_df)
        except:
            print(f"[WARNING] Could not fetch benchmark {benchmark_ticker}")
    
    # Create figure with 2 subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 1], hspace=0.05)
    
    ax1 = fig.add_subplot(gs[0])  # Main price chart
    ax2 = fig.add_subplot(gs[1], sharex=ax1)  # Indicators
    
    # ===== MAIN CHART =====
    plot_momentum_main_chart(ax1, df, ticker, timeframe, demand_zones, supply_zones)
    
    # ===== INDICATORS PANEL =====
    plot_momentum_indicators(ax2, df, rs_data, benchmark_ticker)
    
    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Hide x-axis labels for upper plot
    plt.setp(ax1.get_xticklabels(), visible=False)
    
    # Save plot
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime('%Y-%m-%d')
    filename = f"{ticker.replace('/', '_')}_{timeframe}_momentum_{date_str}.png"
    filepath = Path(save_dir) / filename
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[SUCCESS] Momentum plot saved: {filepath}")
    return filepath


def plot_momentum_main_chart(ax, df, ticker, timeframe, demand_zones, supply_zones):
    """Plot main momentum chart with EMAs and zones"""
    
    # Plot candlesticks
    for idx, row in df.iterrows():
        color = '#26a69a' if row['close'] >= row['open'] else '#ef5350'
        
        # Candle body
        height = abs(row['close'] - row['open'])
        bottom = min(row['open'], row['close'])
        rect = Rectangle((mdates.date2num(idx), bottom), 0.4, height,
                        facecolor=color, edgecolor=color, alpha=0.9)
        ax.add_patch(rect)
        
        # Wicks
        ax.plot([mdates.date2num(idx), mdates.date2num(idx)],
               [row['low'], row['high']], color=color, linewidth=1, alpha=0.9)
    
    # Plot 12/21 Weekly EMAs prominently
    if 'ema_12_weekly' in df.columns:
        ax.plot(df.index, df['ema_12_weekly'], label='12 EMA (Weekly)', 
               color='#2196F3', linewidth=3, alpha=0.9)
    if 'ema_21_weekly' in df.columns:
        ax.plot(df.index, df['ema_21_weekly'], label='21 EMA (Weekly)', 
               color='#9C27B0', linewidth=3, alpha=0.9)
    
    # Monthly opens as thick lines (different colors per month)
    monthly_opens = df.groupby(df.index.to_period('M'))['monthly_open'].first()
    colors_monthly = plt.cm.tab10(np.linspace(0, 1, len(monthly_opens)))
    
    for (period, open_val), color in zip(monthly_opens.items(), colors_monthly):
        period_start = period.to_timestamp()
        period_end = period_start + pd.DateOffset(months=1)
        mask = (df.index >= period_start) & (df.index < period_end)
        if mask.any():
            ax.hlines(open_val, df[mask].index[0], df[mask].index[-1], 
                     colors=color, linewidth=3, alpha=0.7, 
                     label=f'M/O {period_start.strftime("%b %Y")}')
    
    # Quarterly opens as dashed lines
    current_quarterly = df['quarterly_open'].iloc[-1]
    ax.axhline(current_quarterly, color='darkgreen', linestyle='--', linewidth=2.5, 
              alpha=0.7, label='Quarterly Open')
    
    # Demand zones (green boxes)
    for zone in demand_zones:
        start_time = zone['timestamp']
        end_time = df.index[-1]
        width = mdates.date2num(end_time) - mdates.date2num(start_time)
        height = zone['price_high'] - zone['price_low']
        
        rect = FancyBboxPatch(
            (mdates.date2num(start_time), zone['price_low']),
            width, height,
            boxstyle="round,pad=0.01",
            edgecolor='green', facecolor='green',
            alpha=0.15, linewidth=2
        )
        ax.add_patch(rect)
        
        # Label
        ax.text(mdates.date2num(start_time), zone['price_low'], 
               'DEMAND', fontsize=8, color='green', weight='bold',
               verticalalignment='top')
    
    # Supply zones (red boxes)
    for zone in supply_zones:
        start_time = zone['timestamp']
        end_time = df.index[-1]
        width = mdates.date2num(end_time) - mdates.date2num(start_time)
        height = zone['price_high'] - zone['price_low']
        
        rect = FancyBboxPatch(
            (mdates.date2num(start_time), zone['price_low']),
            width, height,
            boxstyle="round,pad=0.01",
            edgecolor='red', facecolor='red',
            alpha=0.15, linewidth=2
        )
        ax.add_patch(rect)
        
        # Label
        ax.text(mdates.date2num(start_time), zone['price_high'], 
               'SUPPLY', fontsize=8, color='red', weight='bold',
               verticalalignment='bottom')
    
    # Identify pullback entry zones (circles at M/O retests)
    monthly_retest_mask = (
        (df['low'] <= df['monthly_open'] * 1.01) & 
        (df['low'] >= df['monthly_open'] * 0.99) &
        (df['close'] > df['monthly_open'])
    )
    ax.scatter(df[monthly_retest_mask].index, df[monthly_retest_mask]['low'],
              marker='o', s=100, color='cyan', edgecolor='blue', linewidth=2,
              label='M/O Retest Entry', zorder=5)
    
    # Breakout points (arrows when clearing resistance with volume)
    if 'volume_threshold' in df.columns:
        resistance = df['high'].rolling(window=20).max()
        breakout_mask = (
            (df['close'] > resistance.shift(1)) &
            (df['volume'] > df['volume_threshold'])
        )
        ax.scatter(df[breakout_mask].index, df[breakout_mask]['high'],
                  marker='^', s=150, color='lime', edgecolor='darkgreen', linewidth=2,
                  label='Breakout', zorder=5)
    
    # Current price annotation
    current_price = df['close'].iloc[-1]
    ax.text(0.02, 0.98, f'Price: ${current_price:.2f}', 
           transform=ax.transAxes, fontsize=14, weight='bold',
           verticalalignment='top', 
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Risk/Reward labels
    if 'ema_12_weekly' in df.columns:
        current_ema12 = df['ema_12_weekly'].iloc[-1]
        risk = abs(current_price - current_ema12)
        
        # Target: next resistance or 3R
        target = current_price + (risk * 3)
        
        ax.text(0.02, 0.90, f'Risk: ${risk:.2f}\nTarget: ${target:.2f}\nR/R: 3.0', 
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top', 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.set_ylabel('Price', fontsize=12, weight='bold')
    ax.set_title(f'{ticker} - Momentum Strategy ({timeframe})', 
                fontsize=14, weight='bold', pad=10)
    ax.legend(loc='upper left', fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)


def plot_momentum_indicators(ax, df, rs_data, benchmark_ticker):
    """Plot momentum indicators panel"""
    
    # Volume bars with threshold
    colors = ['#26a69a' if row['close'] >= row['open'] else '#ef5350' 
              for _, row in df.iterrows()]
    
    ax.bar(df.index, df['volume'], color=colors, alpha=0.5, width=0.4, label='Volume')
    
    # 3x average volume threshold
    if 'volume_threshold' in df.columns:
        ax.axhline(df['volume_threshold'].iloc[-1], color='red', linestyle='--', 
                  linewidth=2, alpha=0.7, label='3x Avg Volume')
    
    # Highlight high volume bars
    high_vol_mask = df['volume'] > df['volume_avg'] * 3
    ax.bar(df[high_vol_mask].index, df[high_vol_mask]['volume'],
          color='yellow', alpha=0.9, width=0.4, edgecolor='red', linewidth=2)
    
    # ATR overlay (secondary axis)
    ax2 = ax.twinx()
    ax2.plot(df.index, df['atr'], color='purple', linewidth=2, label='ATR', alpha=0.7)
    ax2.fill_between(df.index, 0, df['atr'], alpha=0.1, color='purple')
    
    # ATR expansion/compression zones
    atr_ma = df['atr'].rolling(window=20).mean()
    ax2.plot(df.index, atr_ma, color='orange', linestyle='--', linewidth=1.5, 
            label='ATR MA', alpha=0.6)
    
    # Relative strength overlay if available
    if rs_data is not None:
        ax3 = ax.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.plot(df.index, rs_data, color='cyan', linewidth=2, 
                label=f'RS vs {benchmark_ticker}', alpha=0.8)
        ax3.set_ylabel(f'Relative Strength', fontsize=10)
        ax3.legend(loc='upper right', fontsize=8)
    
    ax.set_ylabel('Volume', fontsize=10, weight='bold')
    ax2.set_ylabel('ATR', fontsize=10)
    ax.set_xlabel('Date', fontsize=10, weight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='center left', fontsize=8)
    ax.grid(True, alpha=0.3)

