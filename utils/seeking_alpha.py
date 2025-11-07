import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

# Seeking Alpha API Configuration (RapidAPI)
RAPIDAPI_KEY = "301a4fb04emshc3a8951e1a63c4ep19ba19jsn20a9ad345bb7"
RAPIDAPI_HOST = "seeking-alpha.p.rapidapi.com"
BASE_URL = f"https://{RAPIDAPI_HOST}"

# Cache directory
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)


def _get_cache_path(ticker: str, function: str) -> Path:
    """Get cache file path for a ticker and function."""
    return CACHE_DIR / f"{ticker}_{function}_sa.json"


def _is_cache_valid(cache_path: Path, max_age_hours: int = 24) -> bool:
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
    # This means we only fetch once per day, saving API calls
    return cache_date == current_date


def _get_symbol_id(ticker: str) -> str:
    """
    Get Seeking Alpha symbol ID for a ticker.
    For now, we'll use the ticker directly and let the API handle it.
    In production, you might want to maintain a mapping or use a lookup endpoint.
    """
    return ticker


def _check_cache_needs_update(ticker: str, cache_path: Path, function: str) -> bool:
    """
    Check if cached data needs updating by comparing latest date in cache
    with latest available date from API.
    
    Returns True if cache needs update (new data available), False otherwise.
    """
    if not cache_path.exists():
        return True
    
    try:
        # Load cached data
        with open(cache_path, 'r') as f:
            cached_data = json.load(f)
        
        if not cached_data or ('data' not in cached_data and 'attributes' not in cached_data):
            return True
        
        # Extract latest date from cached data
        cached_latest_date = None
        
        if 'attributes' in cached_data and cached_data['attributes']:
            # Chart data - dates are keys in attributes
            date_strings = list(cached_data['attributes'].keys())
            if date_strings:
                try:
                    dates = [pd.to_datetime(d) for d in date_strings]
                    cached_latest_date = max(dates).date()
                except:
                    pass
        elif 'data' in cached_data and cached_data['data']:
            # For realtime_quotes or other data structures, check if we can extract dates
            # If not, we'll handle it in the function-specific logic below
            pass
        
        # For realtime_quotes, we can't easily determine dates, so we'll handle it separately
        if function == "realtime_quotes":
            # For realtime quotes, always check for updates if cache is from today
            # (realtime data changes frequently)
            return True
        
        if cached_latest_date is None:
            # Can't determine date for chart data, assume needs update
            return True
        
        current_date = datetime.now().date()
        
        # If cached data is not up to today, we need to update
        if cached_latest_date < current_date:
            print(f"  [UPDATE] {ticker} ({function}) cache is outdated: cache={cached_latest_date}, current={current_date}")
            return True
        
        # If cached data is up to today, check API to see if there's newer data
        # (e.g., if data was pulled before market close and now there's EOD data)
        try:
            # Make a small API call to check for updates
            headers = {
                "x-rapidapi-key": RAPIDAPI_KEY,
                "x-rapidapi-host": RAPIDAPI_HOST
            }
            
            if function in ["chart_1y", "chart_5y", "chart_max"]:
                url = f"{BASE_URL}/symbols/get-chart"
                period_map = {"chart_1y": "1y", "chart_5y": "5y", "chart_max": "max"}
                period = period_map.get(function, "1y")
                check_params = {
                    'symbol': ticker.upper(),
                    'period': period
                }
                
                response = requests.get(url, headers=headers, params=check_params)
                
                if response.status_code == 200:
                    api_data = response.json()
                    
                    if 'attributes' in api_data and api_data['attributes']:
                        # Get latest date from API
                        api_date_strings = list(api_data['attributes'].keys())
                        if api_date_strings:
                            try:
                                api_dates = [pd.to_datetime(d) for d in api_date_strings]
                                api_latest_date = max(api_dates).date()
                                
                                # If API has newer data than cache, we need to update
                                if api_latest_date > cached_latest_date:
                                    print(f"  [UPDATE] {ticker} ({function}) has new data: cache={cached_latest_date}, API={api_latest_date}")
                                    return True
                            except:
                                pass
                
                # Small delay to be respectful to API
                time.sleep(1)
            
            return False
            
        except Exception as e:
            # If we can't check, assume cache is still valid
            print(f"  [WARNING] Could not check for updates for {ticker} ({function}): {e}")
            return False
            
    except Exception as e:
        print(f"  [WARNING] Error checking cache update for {ticker} ({function}): {e}")
        return True  # If we can't check, fetch new data to be safe


def _fetch_with_cache(ticker: str, function: str, **params) -> dict:
    """Fetch data from Seeking Alpha API with caching."""
    cache_path = _get_cache_path(ticker, function)
    
    # Try to load from cache
    if _is_cache_valid(cache_path):
        # Cache exists and is from today, but check if new data is available
        if _check_cache_needs_update(ticker, cache_path, function):
            # New data available, need to fetch
            print(f"  [UPDATE] {ticker} ({function}) has new data available, fetching...")
        else:
            # Load from cache (no new data available)
            print(f"Loading {ticker} ({function}) from cache (up to date)...")
            try:
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                    # Validate cached data has expected keys (either 'data' or 'attributes')
                    if cached_data and ('data' in cached_data or 'attributes' in cached_data):
                        return cached_data
                    else:
                        print(f"Cache for {ticker} is invalid, re-fetching...")
                        try:
                            cache_path.unlink(missing_ok=True)
                        except:
                            pass  # Ignore file lock errors
            except Exception as e:
                print(f"  [WARNING] Error reading cache for {ticker}: {e}")
                # Continue to fetch from API
    
    # Fetch from API
    print(f"Fetching {ticker} ({function}) from Seeking Alpha API...")
    
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST
    }
    
    # Determine endpoint based on function
    if function == "realtime_quotes":
        url = f"{BASE_URL}/market/get-realtime-quotes"
        params['sa_ids'] = _get_symbol_id(ticker)
    elif function in ["chart_1y", "chart_5y", "chart_max"]:
        url = f"{BASE_URL}/symbols/get-chart"
        # Extract period from function name
        period_map = {"chart_1y": "1y", "chart_5y": "5y", "chart_max": "max"}
        period = period_map.get(function, "1y")
        params.update({
            'symbol': ticker.upper(),
            'period': period
        })
    else:
        raise ValueError(f"Unsupported function: {function}")
    
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code != 200:
        raise Exception(f"API request failed with status {response.status_code}: {response.text}")
    
    data = response.json()
    
    # Check for API errors
    if 'error' in data or 'message' in data:
        raise Exception(f"Seeking Alpha API error: {data.get('message', data.get('error', 'Unknown error'))}")
    
    # Save valid data to cache
    if data:
        with open(cache_path, 'w') as f:
            json.dump(data, f)
    
    # Rate limiting: be respectful to the API (500 requests/day limit)
    time.sleep(2)  # Increased delay to be more conservative with API limits
    
    return data


def fetch_realtime_quote(ticker: str) -> dict:
    """
    Fetch real-time quote data for a ticker from Seeking Alpha.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary with real-time quote data
    """
    return _fetch_with_cache(ticker, 'realtime_quotes')


def _convert_seeking_alpha_chart_to_df(data: dict) -> pd.DataFrame:
    """Convert Seeking Alpha chart data to DataFrame."""
    if 'attributes' not in data:
        raise Exception("No chart data available in response")
    
    attributes = data['attributes']
    
    if not attributes:
        raise Exception("No historical data available")
    
    # Convert nested dict to records
    records = []
    for date_str, values in attributes.items():
        record = {'Date': date_str}
        record.update(values)
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # Rename columns to match yfinance format
    column_mapping = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }
    
    existing_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=existing_columns)
    
    # Convert Date to datetime and set as index
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
    
    # Select only OHLCV columns
    available_cols = [col for col in ['Open', 'High', 'Low', 'Close', 'Volume'] if col in df.columns]
    df = df[available_cols]
    
    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sort by date (oldest first)
    df = df.sort_index()
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df


def fetch_daily_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Fetch daily data for a ticker from Seeking Alpha.
    
    Args:
        ticker: Stock ticker symbol
        period: Time period (e.g., "1y", "2y", "5y", "max")
    
    Returns:
        DataFrame with OHLCV data matching yfinance format
    """
    try:
        # Map period to chart function
        if period in ["1y", "1yr"]:
            function = 'chart_1y'
        elif period in ["5y", "5yr"]:
            function = 'chart_5y'
        elif period == "max":
            function = 'chart_max'
        else:
            # Default to 1y for 2y or other periods
            function = 'chart_1y'
        
        data = _fetch_with_cache(ticker, function)
        df = _convert_seeking_alpha_chart_to_df(data)
        
        if df.empty:
            raise Exception(f"No data available for {ticker}")
        
        return df
        
    except Exception as e:
        print(f"Error fetching {ticker} daily data: {e}")
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])


def fetch_weekly_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """
    Fetch weekly data for a ticker from Seeking Alpha.
    
    Args:
        ticker: Stock ticker symbol
        period: Time period (e.g., "1y", "2y", "5y", "max")
    
    Returns:
        DataFrame with OHLCV data matching yfinance format
    """
    try:
        data = _fetch_with_cache(ticker, 'historical_week')
        df = _convert_seeking_alpha_to_df(data)
        
        if df.empty:
            raise Exception(f"No data available for {ticker}")
        
        # Filter by period
        if period != "max":
            years = int(period.rstrip('y'))
            cutoff_date = datetime.now() - timedelta(days=years * 365)
            df = df[df.index >= cutoff_date]
        
        return df
        
    except Exception as e:
        print(f"Error fetching {ticker} weekly data: {e}")
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])


def download(tickers, period="1y", interval="1d", group_by="ticker", auto_adjust=True):
    """
    Download market data for multiple tickers (mimics yfinance.download interface).
    
    Args:
        tickers: List of ticker symbols or single ticker string
        period: Time period (e.g., "1y", "2y", "5y")
        interval: Time interval ("1d" for daily, "1wk" for weekly)
        group_by: How to organize multi-ticker data (only "ticker" supported)
        auto_adjust: Whether to use adjusted prices
    
    Returns:
        DataFrame or multi-level DataFrame matching yfinance format
    """
    # Handle single ticker vs multiple tickers
    if isinstance(tickers, str):
        tickers = [tickers]
    
    # Determine which function to use based on interval
    if interval == "1d":
        fetch_func = fetch_daily_data
    elif interval == "1wk":
        fetch_func = fetch_weekly_data
    else:
        raise ValueError(f"Unsupported interval: {interval}. Use '1d' or '1wk'")
    
    # Download data for each ticker
    data_frames = {}
    for i, ticker in enumerate(tickers):
        try:
            df = fetch_func(ticker, period)
            data_frames[ticker] = df
            
            # Add delay between tickers to avoid rate limits
            if i < len(tickers) - 1:
                time.sleep(2)  # Wait 2 seconds between tickers
                
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            # Create empty DataFrame with expected structure
            data_frames[ticker] = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
    
    # If single ticker, return simple DataFrame
    if len(tickers) == 1:
        return data_frames[tickers[0]]
    
    # For multiple tickers, create multi-level columns like yfinance
    if group_by == "ticker":
        # Find all unique dates across all tickers
        all_dates = pd.DatetimeIndex([])
        for df in data_frames.values():
            if not df.empty:
                all_dates = all_dates.union(df.index)
        
        # Create multi-level columns with ticker as first level (matching yfinance format)
        result = pd.DataFrame(index=sorted(all_dates))
        for ticker in tickers:
            for column in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if column in data_frames[ticker].columns and not data_frames[ticker].empty:
                    result[(ticker, column)] = data_frames[ticker][column]
                else:
                    result[(ticker, column)] = pd.Series(dtype=float, index=result.index)
        
        # Convert to proper MultiIndex columns (ticker, column) - same as yfinance
        result.columns = pd.MultiIndex.from_tuples(result.columns, names=['Ticker', 'Price'])
        
        return result
    
    raise ValueError(f"Unsupported group_by: {group_by}")

