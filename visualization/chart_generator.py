"""Chart generation for trading signals with technical indicators"""

# Set matplotlib backend to non-interactive before importing mplfinance
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter issues

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mplfinance as mpf
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple
from core.models import MarketData, TradingSignal
from core.enums import SignalType


class ChartGenerator:
    """Generate candlestick charts with technical indicators"""
    
    def __init__(self, output_dir: str = "plots"):
        """
        Initialize chart generator
        
        Args:
            output_dir: Base directory for saving charts
        """
        self.output_dir = Path(output_dir)
        self.date_str = datetime.now().strftime("%Y-%m-%d")
        self.charts_dir = self.output_dir / self.date_str
        self.charts_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_all_charts(self, market_data: MarketData, signal: TradingSignal):
        """
        Generate both interactive and static charts for a ticker
        
        Args:
            market_data: Market data with historical prices and indicators
            signal: Trading signal with analysis results
        """
        ticker = market_data.ticker
        print(f"  [CHART] Generating charts for {ticker}...", end=" ")
        
        try:
            # Generate interactive Plotly chart
            html_path = self._generate_plotly_chart(market_data, signal)
            
            # Generate static mplfinance chart
            png_path = self._generate_mplfinance_chart(market_data, signal)
            
            print(f"[OK] {html_path.name}, {png_path.name}")
            return html_path, png_path
            
        except Exception as e:
            print(f"[ERROR] {e}")
            return None, None
    
    def _prepare_data(self, market_data: MarketData, lookback: int = 100) -> pd.DataFrame:
        """
        Prepare data for charting (last N days)
        
        Args:
            market_data: Market data object
            lookback: Number of days to include in chart
            
        Returns:
            DataFrame with OHLCV and indicators
        """
        df = market_data.historical.copy()
        
        # Get last N days
        if len(df) > lookback:
            df = df.tail(lookback)
        
        return df
    
    def _generate_plotly_chart(self, market_data: MarketData, 
                               signal: TradingSignal) -> Path:
        """
        Generate interactive Plotly candlestick chart with all indicators
        
        Args:
            market_data: Market data with historical prices
            signal: Trading signal for annotation
            
        Returns:
            Path to saved HTML file
        """
        df = self._prepare_data(market_data)
        ticker = market_data.ticker
        
        # Create subplots: Main chart, Volume, RSI, MACD
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.15, 0.15, 0.2],
            subplot_titles=(
                f'{ticker} - {market_data.short_name or ticker}',
                'Volume',
                'RSI',
                'MACD'
            )
        )
        
        # 1. Main Candlestick Chart
        fig.add_trace(
            go.Candlestick(
                x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=1, col=1
        )
        
        # 2. Bollinger Bands
        if 'bb_upper' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['Date'], y=df['bb_upper'],
                    name='BB Upper',
                    line=dict(color='rgba(173, 204, 255, 0.5)', width=1),
                    showlegend=True
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df['Date'], y=df['bb_mid'],
                    name='BB Mid',
                    line=dict(color='rgba(173, 204, 255, 0.8)', width=1, dash='dash'),
                    showlegend=True
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df['Date'], y=df['bb_lower'],
                    name='BB Lower',
                    line=dict(color='rgba(173, 204, 255, 0.5)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(173, 204, 255, 0.1)',
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # 3. Moving Averages
        if 'ma50' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['Date'], y=df['ma50'],
                    name='MA50',
                    line=dict(color='orange', width=1.5),
                    showlegend=True
                ),
                row=1, col=1
            )
        
        if 'ma200' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['Date'], y=df['ma200'],
                    name='MA200',
                    line=dict(color='purple', width=1.5),
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # 4. Add Signal Annotation on main chart
        signal_color = {
            SignalType.BUY: 'green',
            SignalType.SELL: 'red',
            SignalType.HOLD: 'gray'
        }
        
        last_date = df['Date'].iloc[-1]
        last_close = df['Close'].iloc[-1]
        
        fig.add_annotation(
            x=last_date,
            y=last_close,
            text=f"{signal.signal.value}<br>{signal.confidence_percent:.1f}%",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=signal_color[signal.signal],
            ax=40,
            ay=-40,
            bgcolor=signal_color[signal.signal],
            opacity=0.8,
            font=dict(color='white', size=12),
            row=1, col=1
        )
        
        # 5. Add Entry Zone if available
        if signal.entry_zone:
            fig.add_hrect(
                y0=signal.entry_zone.lower_bound,
                y1=signal.entry_zone.upper_bound,
                line_width=0,
                fillcolor=signal_color[signal.signal],
                opacity=0.1,
                row=1, col=1
            )
        
        # 6. Mark Candlestick Patterns
        self._add_candlestick_patterns(fig, df, row=1)
        
        # 7. Volume Chart
        colors = ['red' if row['Close'] < row['Open'] else 'green' 
                  for _, row in df.iterrows()]
        
        fig.add_trace(
            go.Bar(
                x=df['Date'],
                y=df['Volume'],
                name='Volume',
                marker_color=colors,
                showlegend=False
            ),
            row=2, col=1
        )
        
        if 'volume_sma20' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['Date'], y=df['volume_sma20'],
                    name='Vol SMA20',
                    line=dict(color='blue', width=1),
                    showlegend=True
                ),
                row=2, col=1
            )
        
        # 8. RSI Chart
        if 'rsi' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['Date'], y=df['rsi'],
                    name='RSI',
                    line=dict(color='purple', width=2),
                    showlegend=False
                ),
                row=3, col=1
            )
            
            # Add RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", 
                         opacity=0.5, row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", 
                         opacity=0.5, row=3, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", 
                         opacity=0.3, row=3, col=1)
        
        # 9. MACD Chart
        if 'macd' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['Date'], y=df['macd'],
                    name='MACD',
                    line=dict(color='blue', width=2),
                    showlegend=True
                ),
                row=4, col=1
            )
            
            if 'macd_signal' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['Date'], y=df['macd_signal'],
                        name='Signal',
                        line=dict(color='orange', width=2),
                        showlegend=True
                    ),
                    row=4, col=1
                )
            
            if 'macd_hist' in df.columns:
                colors_macd = ['green' if val >= 0 else 'red' 
                              for val in df['macd_hist']]
                fig.add_trace(
                    go.Bar(
                        x=df['Date'],
                        y=df['macd_hist'],
                        name='Histogram',
                        marker_color=colors_macd,
                        showlegend=True
                    ),
                    row=4, col=1
                )
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                         opacity=0.5, row=4, col=1)
        
        # Update layout
        fig.update_layout(
            title=f"{ticker} - Technical Analysis ({self.date_str})",
            xaxis_rangeslider_visible=False,
            height=1200,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='x unified',
            template='plotly_white'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=4, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
        fig.update_yaxes(title_text="MACD", row=4, col=1)
        
        # Save to HTML
        html_path = self.charts_dir / f"{ticker}_interactive.html"
        fig.write_html(str(html_path))
        
        return html_path
    
    def _add_candlestick_patterns(self, fig, df: pd.DataFrame, row: int):
        """Add candlestick pattern markers to the chart"""
        pattern_columns = [
            'bull_engulfing', 'bear_engulfing',
            'bull_harami', 'bear_harami',
            'bull_morning_star', 'bear_evening_star',
            'bull_3_white', 'bear_3_black',
            'bull_tweezer', 'bear_tweezer',
            'bull_rising_sun', 'bear_dark_cloud'
        ]
        
        for pattern in pattern_columns:
            if pattern not in df.columns:
                continue
            
            # Find where pattern is detected (non-zero values)
            pattern_dates = df[df[pattern] != 0]['Date']
            pattern_prices = df[df[pattern] != 0]['High'] * 1.01  # Slightly above high
            
            if len(pattern_dates) > 0:
                is_bullish = 'bull' in pattern
                fig.add_trace(
                    go.Scatter(
                        x=pattern_dates,
                        y=pattern_prices,
                        mode='markers',
                        name=pattern.replace('_', ' ').title(),
                        marker=dict(
                            symbol='triangle-up' if is_bullish else 'triangle-down',
                            size=10,
                            color='green' if is_bullish else 'red',
                            line=dict(color='white', width=1)
                        ),
                        showlegend=True
                    ),
                    row=row, col=1
                )
    
    def _generate_mplfinance_chart(self, market_data: MarketData, 
                                   signal: TradingSignal) -> Path:
        """
        Generate static mplfinance candlestick chart
        
        Args:
            market_data: Market data with historical prices
            signal: Trading signal for annotation
            
        Returns:
            Path to saved PNG file
        """
        df = self._prepare_data(market_data)
        ticker = market_data.ticker
        
        # Prepare data in mplfinance format
        df_plot = df.set_index('Date')
        df_plot.index = pd.to_datetime(df_plot.index)
        
        # Prepare additional plots
        apds = []
        
        # Add Moving Averages
        if 'ma50' in df_plot.columns and 'ma200' in df_plot.columns:
            apds.append(
                mpf.make_addplot(df_plot['ma50'], color='orange', width=1.5, 
                                label='MA50', panel=0)
            )
            apds.append(
                mpf.make_addplot(df_plot['ma200'], color='purple', width=1.5,
                                label='MA200', panel=0)
            )
        
        # Add Bollinger Bands
        if all(col in df_plot.columns for col in ['bb_upper', 'bb_mid', 'bb_lower']):
            apds.extend([
                mpf.make_addplot(df_plot['bb_upper'], color='lightblue', 
                                width=1, alpha=0.5, panel=0),
                mpf.make_addplot(df_plot['bb_mid'], color='blue', 
                                width=1, linestyle='--', alpha=0.7, panel=0),
                mpf.make_addplot(df_plot['bb_lower'], color='lightblue', 
                                width=1, alpha=0.5, panel=0)
            ])
        
        # Add RSI
        if 'rsi' in df_plot.columns:
            apds.append(
                mpf.make_addplot(df_plot['rsi'], color='purple', width=2,
                                ylabel='RSI', panel=2, ylim=(0, 100))
            )
        
        # Add MACD
        if 'macd' in df_plot.columns and 'macd_signal' in df_plot.columns:
            apds.extend([
                mpf.make_addplot(df_plot['macd'], color='blue', width=2,
                                panel=3, ylabel='MACD'),
                mpf.make_addplot(df_plot['macd_signal'], color='orange', 
                                width=2, panel=3)
            ])
            
            if 'macd_hist' in df_plot.columns:
                colors = ['green' if v >= 0 else 'red' for v in df_plot['macd_hist']]
                apds.append(
                    mpf.make_addplot(df_plot['macd_hist'], type='bar',
                                    color=colors, panel=3, alpha=0.5)
                )
        
        # Style configuration
        signal_color = {
            SignalType.BUY: 'green',
            SignalType.SELL: 'red',
            SignalType.HOLD: 'gray'
        }
        
        mc = mpf.make_marketcolors(
            up='#26a69a', down='#ef5350',
            edge='inherit',
            wick='inherit',
            volume='in'
        )
        
        s = mpf.make_mpf_style(
            marketcolors=mc,
            gridstyle='--',
            y_on_right=False
        )
        
        # Create title with signal
        title = (f"{ticker} - {market_data.short_name or ticker}\n"
                f"Signal: {signal.signal.value} ({signal.confidence_percent:.1f}%)")
        
        # Save to PNG
        png_path = self.charts_dir / f"{ticker}_static.png"
        
        mpf.plot(
            df_plot,
            type='candle',
            style=s,
            title=title,
            volume=True,
            addplot=apds if apds else None,
            figsize=(16, 12),
            panel_ratios=(3, 1, 1, 1.5),
            savefig=dict(fname=str(png_path), dpi=150, bbox_inches='tight')
        )
        
        return png_path
    
    def generate_summary_report(self, signals: List[TradingSignal], market: str):
        """
        Generate a summary HTML report with links to all charts
        
        Args:
            signals: List of trading signals
            market: Market name (US, SG, etc.)
        """
        report_path = self.charts_dir / f"{market.lower()}_report.html"
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{market} Market Analysis - {self.date_str}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
        }}
        .signal-card {{
            background: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .buy {{ border-left: 5px solid green; }}
        .sell {{ border-left: 5px solid red; }}
        .hold {{ border-left: 5px solid gray; }}
        .ticker {{
            font-size: 1.2em;
            font-weight: bold;
        }}
        .confidence {{
            color: #666;
        }}
        a {{
            color: #0066cc;
            text-decoration: none;
            margin-right: 15px;
        }}
        a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <h1>{market} Market Analysis Report</h1>
    <p>Generated: {self.date_str}</p>
    <p>Total Tickers: {len(signals)}</p>
"""
        
        for signal in signals:
            signal_class = signal.signal.value.lower()
            html_content += f"""
    <div class="signal-card {signal_class}">
        <div class="ticker">{signal.ticker} - {signal.short_name}</div>
        <div class="confidence">Signal: {signal.signal.value} | Confidence: {signal.confidence_percent:.1f}%</div>
        <div>
            <a href="{signal.ticker}_interactive.html" target="_blank">Interactive Chart</a>
            <a href="{signal.ticker}_static.png" target="_blank">Static Chart</a>
        </div>
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"[SUCCESS] Summary report saved: {report_path}")
        return report_path

