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
            title=dict(
                text=f"{ticker} - Technical Analysis ({self.date_str})",
                x=0.5,
                xanchor='center',
                font=dict(size=16)
            ),
            xaxis_rangeslider_visible=False,
            height=1200,
            autosize=True,
            width=None,  # Let it be responsive
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.12,  # Position below the chart
                xanchor="center",
                x=0.5,
                font=dict(size=9),
                itemsizing='constant'
            ),
            margin=dict(
                t=80,  # Top margin for title
                b=120,  # Bottom margin for legend
                l=60,
                r=60
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
        
        # Save to HTML with responsive config
        html_path = self.charts_dir / f"{ticker}_interactive.html"
        fig.write_html(
            str(html_path),
            config={
                'responsive': True,
                'displayModeBar': True,
                'displaylogo': False
            },
            include_plotlyjs='cdn'
        )
        
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
        html, body {{
            height: 100%;
            margin: 0;
            padding: 0;
            overflow-y: auto;
            overflow-x: hidden;
        }}
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
    
    def generate_index_html(self, root_dir: str = "."):
        """
        Generate an index.html file in the root directory that links to US and SG reports
        for GitHub Pages deployment. Scans all available date folders in plots/ directory.
        
        Args:
            root_dir: Root directory where index.html should be created
        """
        index_path = Path(root_dir) / "index.html"
        plots_dir = Path(root_dir) / "plots"
        crypto_plots_dir = Path(root_dir) / "traderXO" / "plots"
        
        # Find all available date folders
        available_dates = []
        if plots_dir.exists():
            for date_dir in sorted(plots_dir.iterdir(), reverse=True):
                if date_dir.is_dir() and date_dir.name.startswith("20"):  # Date folders start with year
                    date_str = date_dir.name
                    us_report = date_dir / "us_report.html"
                    sg_report = date_dir / "sg_report.html"
                    crypto_report = crypto_plots_dir / date_str / "crypto_report.html" if crypto_plots_dir.exists() else None
                    
                    if us_report.exists() or sg_report.exists() or (crypto_report and crypto_report.exists()):
                        available_dates.append({
                            "date": date_str,
                            "us_exists": us_report.exists(),
                            "sg_exists": sg_report.exists(),
                            "crypto_exists": crypto_report.exists() if crypto_report else False,
                            "is_latest": date_str == self.date_str
                        })
        
        # Get latest date (first in sorted list, or current date)
        latest_date = available_dates[0]["date"] if available_dates else self.date_str
        
        # Build HTML for date sections
        date_sections_html = ""
        for date_info in available_dates:
            date_str = date_info["date"]
            report_path = f"plots/{date_str}"
            is_latest = date_info["is_latest"]
            
            date_sections_html += f"""
        <div class="date-section {'latest' if is_latest else ''}">
            <h3 class="date-header">
                📅 {date_str}
                {('<span class="latest-badge">Latest</span>' if is_latest else '')}
            </h3>
            <div class="report-links">
"""
            if date_info["us_exists"]:
                date_sections_html += f"""
                <a href="{report_path}/us_report.html" class="report-card us" target="_blank">
                    <div class="report-title">🇺🇸 US Market Report</div>
                    <div class="report-description">View analysis for US stock market tickers</div>
                </a>
"""
            if date_info["sg_exists"]:
                date_sections_html += f"""
                <a href="{report_path}/sg_report.html" class="report-card sg" target="_blank">
                    <div class="report-title">🇸🇬 Singapore Market Report</div>
                    <div class="report-description">View analysis for Singapore stock market tickers</div>
                </a>
"""
            if date_info.get("crypto_exists", False):
                crypto_report_path = f"traderXO/plots/{date_str}"
                date_sections_html += f"""
                <a href="{crypto_report_path}/crypto_report.html" class="report-card crypto" target="_blank">
                    <div class="report-title">🪙 Crypto Market Report</div>
                    <div class="report-description">View TraderXO crypto analysis charts</div>
                </a>
"""
            date_sections_html += """
            </div>
        </div>
"""
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Market Analysis Reports</title>
    <style>
        html, body {{
            height: 100%;
            margin: 0;
            padding: 0;
            overflow-y: auto;
            overflow-x: hidden;
        }}
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            padding: 40px;
            max-width: 800px;
            margin: 0 auto;
        }}
        h1 {{
            color: #333;
            margin-bottom: 10px;
            font-size: 2.5em;
            text-align: center;
        }}
        .subtitle {{
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
            text-align: center;
        }}
        .date-section {{
            margin-bottom: 40px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            border: 2px solid #e9ecef;
        }}
        .date-section.latest {{
            border-color: #667eea;
            background: #f0f4ff;
        }}
        .date-header {{
            color: #333;
            margin-bottom: 20px;
            font-size: 1.3em;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .latest-badge {{
            background: #667eea;
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.7em;
            font-weight: normal;
        }}
        .report-links {{
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}
        .report-card {{
            background: white;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            padding: 20px;
            transition: all 0.3s ease;
            text-decoration: none;
            color: inherit;
            display: block;
        }}
        .report-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            border-color: #667eea;
        }}
        .report-card.us {{
            border-left: 5px solid #28a745;
        }}
        .report-card.sg {{
            border-left: 5px solid #ffc107;
        }}
        .report-card.crypto {{
            border-left: 5px solid #9c27b0;
        }}
        .report-title {{
            font-size: 1.3em;
            font-weight: bold;
            color: #333;
            margin-bottom: 8px;
        }}
        .report-description {{
            color: #666;
            font-size: 0.95em;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e9ecef;
            color: #999;
            font-size: 0.85em;
            text-align: center;
        }}
        .no-reports {{
            text-align: center;
            color: #666;
            padding: 40px;
            font-size: 1.1em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Market Analysis Reports</h1>
        <p class="subtitle">Trading Signal Analysis Dashboard</p>
        
        {date_sections_html if available_dates else '<div class="no-reports">No reports available yet. Run the analysis to generate reports.</div>'}
        
        <div class="footer">
            <p>Generated automatically by Market Analysis System</p>
            <p>Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}</p>
        </div>
    </div>
</body>
</html>
"""
        
        with open(index_path, 'w') as f:
            f.write(html_content)
        
        print(f"[SUCCESS] Index page saved: {index_path}")
        if available_dates:
            print(f"  Found {len(available_dates)} date folder(s) with reports")
        return index_path
    
    def generate_tabbed_charts_view(self, signals: List[TradingSignal], market: str):
        """
        Generate a single HTML file with tabs to view all stock charts
        
        Args:
            signals: List of trading signals
            market: Market name (US, SG, etc.)
            
        Returns:
            Path to saved HTML file
        """
        report_path = self.charts_dir / f"{market.lower()}_all_charts.html"
        
        # Build tabs HTML
        tabs_html = ""
        tab_content_html = ""
        
        for idx, signal in enumerate(signals):
            ticker = signal.ticker
            html_file = self.charts_dir / f"{ticker}_interactive.html"
            
            # Check if the HTML file exists
            if html_file.exists():
                signal_class = signal.signal.value.lower()
                signal_color = {
                    'buy': '#28a745',
                    'sell': '#dc3545',
                    'hold': '#6c757d'
                }
                
                # Create tab button
                is_active = "active" if idx == 0 else ""
                tabs_html += f"""
                <button class="tab-button {is_active}" onclick="openTab(event, '{ticker}')" 
                        style="border-left: 4px solid {signal_color.get(signal_class, '#6c757d')};">
                    <span class="ticker-name">{ticker}</span>
                    <span class="signal-badge {signal_class}">{signal.signal.value}</span>
                    <span class="confidence">{signal.confidence_percent:.1f}%</span>
                </button>
                """
                
                # Create tab content with iframe
                active_class = "active" if idx == 0 else ""
                tab_content_html += f"""
                <div id="{ticker}" class="tab-content {active_class}">
                    <div class="chart-header">
                        <h2>{ticker} - {signal.short_name}</h2>
                        <div class="signal-info">
                            <span class="signal-label">Signal: <strong class="{signal_class}">{signal.signal.value}</strong></span>
                            <span class="confidence-label">Confidence: <strong>{signal.confidence_percent:.1f}%</strong></span>
                        </div>
                    </div>
                    <iframe src="{ticker}_interactive.html" class="chart-iframe" frameborder="0" scrolling="no"></iframe>
                </div>
                """
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{market} Market Analysis - All Charts - {self.date_str}</title>
    <meta charset="utf-8">
    <style>
        * {{
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f7fa;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 28px;
            font-weight: 600;
        }}
        
        .header p {{
            margin: 5px 0 0 0;
            opacity: 0.9;
            font-size: 14px;
        }}
        
        .container {{
            display: flex;
            height: calc(100vh - 100px);
        }}
        
        .tabs-sidebar {{
            width: 280px;
            background: white;
            border-right: 1px solid #e0e0e0;
            overflow-y: auto;
            box-shadow: 2px 0 5px rgba(0,0,0,0.05);
        }}
        
        .tabs-sidebar::-webkit-scrollbar {{
            width: 6px;
        }}
        
        .tabs-sidebar::-webkit-scrollbar-track {{
            background: #f1f1f1;
        }}
        
        .tabs-sidebar::-webkit-scrollbar-thumb {{
            background: #888;
            border-radius: 3px;
        }}
        
        .tabs-sidebar::-webkit-scrollbar-thumb:hover {{
            background: #555;
        }}
        
        .tab-button {{
            width: 100%;
            padding: 15px 20px;
            text-align: left;
            background: white;
            border: none;
            border-bottom: 1px solid #f0f0f0;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            flex-direction: column;
            gap: 5px;
        }}
        
        .tab-button:hover {{
            background: #f8f9fa;
        }}
        
        .tab-button.active {{
            background: #e3f2fd;
            border-left-width: 6px;
        }}
        
        .ticker-name {{
            font-size: 16px;
            font-weight: 600;
            color: #333;
        }}
        
        .signal-badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .signal-badge.buy {{
            background: #d4edda;
            color: #155724;
        }}
        
        .signal-badge.sell {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        .signal-badge.hold {{
            background: #e2e3e5;
            color: #383d41;
        }}
        
        .confidence {{
            font-size: 12px;
            color: #666;
            font-weight: 500;
        }}
        
        .content-area {{
            flex: 1;
            overflow: hidden;
            position: relative;
        }}
        
        .tab-content {{
            height: 100%;
            display: none;
            overflow: hidden;
            flex-direction: column;
        }}
        
        .tab-content.active {{
            display: flex;
        }}
        
        .chart-header {{
            padding: 20px 30px;
            background: white;
            border-bottom: 1px solid #e0e0e0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            flex-shrink: 0;
        }}
        
        .chart-header h2 {{
            margin: 0 0 10px 0;
            font-size: 24px;
            color: #333;
        }}
        
        .signal-info {{
            display: flex;
            gap: 20px;
            font-size: 14px;
        }}
        
        .signal-label, .confidence-label {{
            color: #666;
        }}
        
        .signal-label strong.buy {{
            color: #28a745;
        }}
        
        .signal-label strong.sell {{
            color: #dc3545;
        }}
        
        .signal-label strong.hold {{
            color: #6c757d;
        }}
        
        .chart-iframe {{
            flex: 1;
            width: 100%;
            border: none;
            background: white;
            min-height: 0;
            overflow: hidden;
            display: block;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                flex-direction: column;
            }}
            
            .tabs-sidebar {{
                width: 100%;
                height: 200px;
                border-right: none;
                border-bottom: 1px solid #e0e0e0;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{market} Market Analysis - All Charts</h1>
        <p>Generated: {self.date_str} | Total Tickers: {len(signals)}</p>
    </div>
    
    <div class="container">
        <div class="tabs-sidebar">
            {tabs_html}
        </div>
        
        <div class="content-area">
            {tab_content_html}
        </div>
    </div>
    
    <script>
        function openTab(evt, tickerName) {{
            var i, tabcontent, tabbuttons;
            
            // Hide all tab content
            tabcontent = document.getElementsByClassName("tab-content");
            for (i = 0; i < tabcontent.length; i++) {{
                tabcontent[i].classList.remove("active");
            }}
            
            // Remove active class from all buttons
            tabbuttons = document.getElementsByClassName("tab-button");
            for (i = 0; i < tabbuttons.length; i++) {{
                tabbuttons[i].classList.remove("active");
            }}
            
            // Show the selected tab and mark button as active
            document.getElementById(tickerName).classList.add("active");
            evt.currentTarget.classList.add("active");
        }}
        
        // Keyboard navigation
        document.addEventListener('keydown', function(e) {{
            var tabs = document.querySelectorAll('.tab-button');
            var activeIndex = Array.from(tabs).findIndex(btn => btn.classList.contains('active'));
            
            if (e.key === 'ArrowDown' || e.key === 'ArrowRight') {{
                e.preventDefault();
                var nextIndex = (activeIndex + 1) % tabs.length;
                tabs[nextIndex].click();
            }} else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') {{
                e.preventDefault();
                var prevIndex = (activeIndex - 1 + tabs.length) % tabs.length;
                tabs[prevIndex].click();
            }}
        }});
    </script>
</body>
</html>
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"[SUCCESS] Tabbed charts view saved: {report_path}")
        return report_path

