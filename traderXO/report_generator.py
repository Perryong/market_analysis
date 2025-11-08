"""
Generate summary reports for TraderXO crypto charts
Similar to the stock market report generator
"""

from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional


class TraderXOReportGenerator:
    """Generate summary HTML reports for TraderXO crypto charts"""
    
    def __init__(self, plots_dir: str = "traderXO/plots"):
        """
        Initialize report generator
        
        Args:
            plots_dir: Base directory for traderXO plots
        """
        self.plots_dir = Path(plots_dir)
        self.date_str = datetime.now().strftime("%Y-%m-%d")
    
    def generate_summary_report(self, date_str: Optional[str] = None) -> Path:
        """
        Generate a summary HTML report for a specific date's crypto charts
        
        Args:
            date_str: Date string in YYYY-MM-DD format. If None, uses current date.
            
        Returns:
            Path to the generated report file
        """
        if date_str is None:
            date_str = self.date_str
        
        date_dir = self.plots_dir / date_str
        
        if not date_dir.exists():
            print(f"[WARNING] No plots directory found for {date_str}")
            return None
        
        # Find all PNG files in the date directory
        plot_files = sorted(date_dir.glob("*.png"))
        
        if not plot_files:
            print(f"[WARNING] No plots found for {date_str}")
            return None
        
        # Organize plots by ticker and strategy
        plots_by_ticker = {}
        for plot_file in plot_files:
            filename = plot_file.stem  # e.g., "BTC_USDT_1d_momentum"
            parts = filename.split('_')
            
            if len(parts) >= 3:
                ticker = f"{parts[0]}/{parts[1]}"  # e.g., "BTC/USDT"
                timeframe = parts[2]  # e.g., "1d"
                strategy = '_'.join(parts[3:]) if len(parts) > 3 else 'unknown'
                
                if ticker not in plots_by_ticker:
                    plots_by_ticker[ticker] = {}
                
                if strategy not in plots_by_ticker[ticker]:
                    plots_by_ticker[ticker][strategy] = []
                
                interactive_name = self._ensure_interactive_html(plot_file)

                plots_by_ticker[ticker][strategy].append({
                    'file': plot_file.name,
                    'timeframe': timeframe,
                    'strategy': strategy,
                    'interactive': interactive_name
                })
        
        # Generate HTML report
        report_path = date_dir / "crypto_report.html"
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Crypto Market Analysis - {date_str}</title>
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
        .ticker-section {{
            background: white;
            padding: 20px;
            margin: 15px 0;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .ticker-header {{
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
            margin-bottom: 15px;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .strategy-group {{
            margin: 15px 0;
        }}
        .strategy-title {{
            font-size: 1.1em;
            font-weight: bold;
            color: #666;
            margin-bottom: 10px;
            text-transform: capitalize;
        }}
        .plot-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
            margin-top: 10px;
        }}
        .plot-card {{
            background: #f8f9fa;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            padding: 15px;
            transition: all 0.3s ease;
        }}
        .plot-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            border-color: #667eea;
        }}
        .plot-card img {{
            width: 100%;
            height: auto;
            border-radius: 5px;
            margin-bottom: 10px;
        }}
        .plot-info {{
            font-size: 0.9em;
            color: #666;
        }}
        .timeframe-badge {{
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.8em;
            margin-right: 5px;
        }}
        .plot-links {{
            margin-top: 10px;
        }}
        .plot-links a {{
            color: #0066cc;
            text-decoration: none;
            margin-right: 15px;
            font-weight: bold;
            font-size: 0.85em;
        }}
        .plot-links a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <h1>🪙 Crypto Market Analysis Report</h1>
    <p>Generated: {date_str}</p>
    <p>Total Tickers: {len(plots_by_ticker)}</p>
"""
        
        # Generate HTML for each ticker
        for ticker in sorted(plots_by_ticker.keys()):
            ticker_data = plots_by_ticker[ticker]
            html_content += f"""
    <div class="ticker-section">
        <div class="ticker-header">{ticker}</div>
"""
            
            # Group by strategy
            for strategy in sorted(ticker_data.keys()):
                plots = ticker_data[strategy]
                strategy_display = strategy.replace('_', ' ').title()
                
                html_content += f"""
        <div class="strategy-group">
            <div class="strategy-title">{strategy_display}</div>
            <div class="plot-grid">
"""
                
                for plot in plots:
                    plot_name = plot['file']
                    timeframe = plot['timeframe']
                    interactive = plot.get('interactive')
                    interactive_link = ""
                    if interactive:
                        interactive_link = f'<a href="{interactive}" target="_blank" rel="noopener">Interactive Chart</a>'
                    static_link = f'<a href="{plot_name}" target="_blank" rel="noopener">Static Chart</a>'
                    html_content += f"""
                <div class="plot-card">
                    <img src="{plot_name}" alt="{ticker} {strategy_display}" loading="lazy">
                    <div class="plot-info">
                        <span class="timeframe-badge">{timeframe}</span>
                        {strategy_display}
                    </div>
                    <div class="plot-links">
                        {interactive_link}
                        {static_link}
                    </div>
                </div>
"""
                
                html_content += """
            </div>
        </div>
"""
            
            html_content += """
    </div>
"""
        
        html_content += """
    </body>
</html>
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"[SUCCESS] Crypto report saved: {report_path}")
        return report_path

    def _ensure_interactive_html(self, png_path: Path) -> Optional[str]:
        """
        Ensure that an interactive HTML viewer exists for the given PNG plot.

        Returns the name of the interactive HTML file if available, otherwise None.
        """
        html_path = png_path.with_name(f"{png_path.stem}_interactive.html")

        try:
            html_content = self._build_interactive_html(
                image_name=png_path.name,
                title=png_path.stem.replace('_', ' ')
            )
            html_path.write_text(html_content, encoding='utf-8')
            return html_path.name
        except Exception as exc:
            print(f"[WARNING] Could not create interactive view for {png_path.name}: {exc}")
            return None

    @staticmethod
    def _build_interactive_html(image_name: str, title: str) -> str:
        """Build a lightweight interactive viewer for a PNG using OpenSeadragon."""
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="preconnect" href="https://cdnjs.cloudflare.com">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/openseadragon.min.js" integrity="sha512-VKBuvrXdP1AXvfs+m4l3ZNZSI4PFJF0K0hGJJZ4RiNRkvFMO4IwFRHkoTc7xsdZhMgkLn+Ioq4elndAZicBcRQ==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
    <style>
        html, body {{
            margin: 0;
            height: 100%;
            background: #0f172a;
            color: #e2e8f0;
            font-family: Arial, sans-serif;
        }}
        header {{
            padding: 12px 24px;
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(6px);
            position: sticky;
            top: 0;
            z-index: 100;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        }}
        header h1 {{
            margin: 0;
            font-size: 1.1rem;
        }}
        header a {{
            color: #93c5fd;
            text-decoration: none;
            font-size: 0.9rem;
        }}
        #viewer {{
            width: 100%;
            height: calc(100% - 58px);
            background: #0f172a;
        }}
        .hint {{
            font-size: 0.85rem;
            color: #94a3b8;
        }}
    </style>
</head>
<body>
    <header>
        <h1>{title}</h1>
        <div class="hint">Use mouse wheel or pinch to zoom · Click and drag to pan · Double-click to zoom</div>
        <a href="{image_name}" download>Download PNG</a>
    </header>
    <div id="viewer"></div>
    <script>
        OpenSeadragon({{
            id: "viewer",
            prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/images/",
            tileSources: {{
                type: "image",
                url: "{image_name}"
            }},
            showRotationControl: true,
            showNavigator: true,
            navigatorAutoFade: false,
            defaultZoomLevel: 1,
            minZoomLevel: 0.5,
            maxZoomLevel: 20,
            zoomPerClick: 1.5,
            gestureSettingsTouch: {{
                pinchToZoom: true,
                flickEnabled: true
            }},
            gestureSettingsMouse: {{
                clickToZoom: true,
                dblClickToZoom: true
            }}
        }});
    </script>
</body>
</html>
"""
    
    def generate_all_reports(self) -> List[Path]:
        """
        Generate reports for all available date directories
        
        Returns:
            List of paths to generated reports
        """
        if not self.plots_dir.exists():
            print(f"[WARNING] Plots directory not found: {self.plots_dir}")
            return []
        
        report_paths = []
        
        # Find all date directories
        for date_dir in sorted(self.plots_dir.iterdir(), reverse=True):
            if date_dir.is_dir() and date_dir.name.startswith("20"):  # Date folders start with year
                try:
                    report_path = self.generate_summary_report(date_dir.name)
                    if report_path:
                        report_paths.append(report_path)
                except Exception as e:
                    print(f"[ERROR] Failed to generate report for {date_dir.name}: {e}")
        
        return report_paths

