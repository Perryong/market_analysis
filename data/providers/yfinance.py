"""yfinance data provider implementation"""

import json
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import yfinance as yf
from typing import List, Dict, Optional
from core.models import MarketData
from core.enums import TimeFrame
from technical.calculator import TechnicalCalculator

# Cache directory for yfinance data
CACHE_DIR = Path(".cache/yfinance")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class YFinanceProvider:
    """Data provider using yfinance with caching"""
    
    def __init__(self):
        self.calculator = TechnicalCalculator()
    
    def _get_cache_path(self, ticker: str, period: str, timeframe: TimeFrame) -> Path:
        """Get cache file path for a ticker"""
        interval = timeframe.value.replace('d', 'day')  # Convert '1d' to '1day' for filename
        return CACHE_DIR / f"{ticker}_{period}_{interval}_yf.json"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """
        Check if cache file exists and is from today (same calendar day).
        This minimizes API calls by reusing data throughout the trading day.
        """
        if not cache_path.exists():
            return False
        
        # Get cache file modification time
        cache_mtime = cache_path.stat().st_mtime
        cache_date = datetime.fromtimestamp(cache_mtime).date()
        
        # Get current date
        current_date = datetime.now().date()
        
        # Cache is valid if it's from today
        return cache_date == current_date
    
    def _load_from_cache(self, cache_path: Path) -> Optional[pd.DataFrame]:
        """Load DataFrame from cache file"""
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
                # Reconstruct DataFrame from JSON
                df = pd.DataFrame(
                    data['data'],
                    index=pd.to_datetime(data['index']),
                    columns=data['columns']
                )
                df.index.name = 'Date'
                return df
        except Exception as e:
            print(f"  [WARNING] Cache read error: {e}")
            return None
    
    def _save_to_cache(self, df: pd.DataFrame, cache_path: Path):
        """Save DataFrame to cache file"""
        try:
            # Convert DataFrame to JSON-serializable format
            cache_data = {
                'index': df.index.strftime('%Y-%m-%d').tolist(),
                'columns': df.columns.tolist(),
                'data': df.values.tolist()
            }
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            print(f"  [WARNING] Cache write error: {e}")
        
    def fetch_data(self, ticker: str, period: str = "2y", 
                   timeframe: TimeFrame = TimeFrame.DAILY) -> pd.DataFrame:
        """
        Fetch OHLCV data from yfinance
        
        Args:
            ticker: Stock ticker symbol
            period: Time period
            timeframe: Data granularity
            
        Returns:
            DataFrame with OHLCV data
        """
        interval = timeframe.value
        
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True
        )
        
        return df
    
    def get_market_data(self, ticker: str, period: str = "2y",
                       timeframe: TimeFrame = TimeFrame.DAILY) -> MarketData:
        """
        Fetch and prepare market data with all indicators
        
        Args:
            ticker: Stock ticker symbol
            period: Time period
            timeframe: Data granularity
            
        Returns:
            Complete MarketData object
        """
        # Fetch raw data
        df = self.fetch_data(ticker, period, timeframe)
        
        if df.empty:
            raise ValueError(f"No data available for {ticker}")
        
        # Calculate all technical indicators
        df = self.calculator.calculate_all(df)
        
        # Store ticker for later reference
        df.attrs['TICKER'] = ticker
        
        # Extract current and previous rows
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Get company name
        try:
            ticker_obj = yf.Ticker(ticker)
            short_name = ticker_obj.info.get("shortName", ticker)
        except:
            short_name = ticker
        
        return MarketData(
            ticker=ticker,
            current=current,
            previous=previous,
            historical=df,
            timeframe=timeframe,
            short_name=short_name
        )
    
    def get_ticker_name(self, ticker: str) -> str:
        """Get human-readable ticker name"""
        try:
            ticker_obj = yf.Ticker(ticker)
            return ticker_obj.info.get("shortName", ticker)
        except:
            return ticker
    
    def fetch_data_batch(self, tickers: List[str], period: str = "2y",
                         timeframe: TimeFrame = TimeFrame.DAILY,
                         max_retries: int = 3,
                         retry_delay: int = 60) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple tickers using batch download with caching
        This helps avoid rate limiting by downloading multiple tickers at once
        
        Args:
            tickers: List of stock ticker symbols
            period: Time period
            timeframe: Data granularity
            max_retries: Maximum number of retry attempts
            retry_delay: Delay in seconds between retries
            
        Returns:
            Dictionary mapping ticker to DataFrame with OHLCV data
        """
        result = {}
        tickers_to_fetch = []
        
        # Check cache for each ticker
        print(f"  [*] Checking cache for {len(tickers)} tickers...")
        for ticker in tickers:
            cache_path = self._get_cache_path(ticker, period, timeframe)
            
            if self._is_cache_valid(cache_path):
                # Load from cache
                df = self._load_from_cache(cache_path)
                if df is not None and not df.empty:
                    result[ticker] = df
                    print(f"  [CACHE] {ticker} loaded from cache")
                else:
                    tickers_to_fetch.append(ticker)
            else:
                tickers_to_fetch.append(ticker)
        
        # If all tickers are cached, return immediately
        if not tickers_to_fetch:
            print(f"  [SUCCESS] All {len(tickers)} tickers loaded from cache")
            return result
        
        # Batch download remaining tickers
        print(f"  [*] Fetching {len(tickers_to_fetch)} tickers from yfinance...")
        interval = timeframe.value
        
        for attempt in range(max_retries):
            try:
                # Download all tickers at once
                print(f"  [*] Batch downloading {len(tickers_to_fetch)} tickers (attempt {attempt + 1}/{max_retries})...")
                
                data = yf.download(
                    tickers_to_fetch,
                    period=period,
                    interval=interval,
                    progress=False,
                    auto_adjust=True,
                    group_by='ticker'
                )
                
                # If only one ticker, yfinance returns a different structure
                if len(tickers_to_fetch) == 1:
                    ticker = tickers_to_fetch[0]
                    if not data.empty:
                        result[ticker] = data
                        # Save to cache
                        cache_path = self._get_cache_path(ticker, period, timeframe)
                        self._save_to_cache(data, cache_path)
                else:
                    # Multiple tickers - extract each one
                    for ticker in tickers_to_fetch:
                        try:
                            if ticker in data.columns.levels[0]:
                                df = data[ticker].copy()
                                if not df.empty and not df.isna().all().all():
                                    result[ticker] = df
                                    # Save to cache
                                    cache_path = self._get_cache_path(ticker, period, timeframe)
                                    self._save_to_cache(df, cache_path)
                        except Exception as e:
                            print(f"  [WARNING] Could not extract data for {ticker}: {e}")
                            continue
                
                fetched_count = len([t for t in tickers_to_fetch if t in result])
                if fetched_count > 0:
                    cached_count = len(tickers) - len(tickers_to_fetch)
                    print(f"  [SUCCESS] Downloaded {fetched_count} tickers, {cached_count} from cache (total: {len(result)}/{len(tickers)})")
                    return result
                
                # If no data, might be rate limited
                if attempt < max_retries - 1:
                    print(f"  [WARNING] No data received, waiting {retry_delay}s before retry...")
                    time.sleep(retry_delay)
                    
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check if it's a rate limit error
                if 'rate limit' in error_msg or 'too many requests' in error_msg:
                    if attempt < max_retries - 1:
                        print(f"  [WARNING] Rate limited. Waiting {retry_delay}s before retry...")
                        time.sleep(retry_delay)
                    else:
                        print(f"  [ERROR] Rate limit exceeded after {max_retries} attempts")
                        raise
                else:
                    # For other errors, raise immediately
                    print(f"  [ERROR] Batch download failed: {e}")
                    raise
        
        return result
    
    def get_market_data_batch(self, tickers: List[str], period: str = "2y",
                              timeframe: TimeFrame = TimeFrame.DAILY) -> List[MarketData]:
        """
        Fetch and prepare market data for multiple tickers with rate limiting
        
        Args:
            tickers: List of stock ticker symbols
            period: Time period
            timeframe: Data granularity
            
        Returns:
            List of MarketData objects
        """
        # Fetch all data at once
        batch_data = self.fetch_data_batch(tickers, period, timeframe)
        
        results = []
        
        for ticker in tickers:
            try:
                print(f"Analyzing {ticker}...", end=" ")
                
                # Check if we got data for this ticker
                if ticker not in batch_data:
                    print(f"[ERROR] No data available for {ticker}")
                    continue
                
                df = batch_data[ticker]
                
                if df.empty:
                    print(f"[ERROR] Empty data for {ticker}")
                    continue
                
                # Calculate all technical indicators
                df = self.calculator.calculate_all(df)
                
                # Store ticker for later reference
                df.attrs['TICKER'] = ticker
                
                # Extract current and previous rows
                current = df.iloc[-1]
                previous = df.iloc[-2]
                
                # Get company name (with retry logic)
                short_name = ticker
                try:
                    ticker_obj = yf.Ticker(ticker)
                    short_name = ticker_obj.info.get("shortName", ticker)
                except:
                    pass  # Use ticker as fallback
                
                market_data = MarketData(
                    ticker=ticker,
                    current=current,
                    previous=previous,
                    historical=df,
                    timeframe=timeframe,
                    short_name=short_name
                )
                
                results.append(market_data)
                print(f"[OK]")
                
            except Exception as e:
                print(f"[ERROR] {e}")
                continue
        
        return results

