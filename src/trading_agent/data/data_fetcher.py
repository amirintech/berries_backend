import pandas as pd
import datetime
import yfinance as yf
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

class DataFetcher:
    """
    Class for fetching data from different sources
    """
    def __init__(self, api_key=None, secret_key=None):
        """
        Initialize with optional Alpaca API credentials
        """
        self.api_key = api_key
        self.secret_key = secret_key
        
        # Initialize Alpaca client if credentials are provided
        self.stock_client = None
        if api_key and secret_key:
            self.stock_client = StockHistoricalDataClient(api_key, secret_key)
    
    def fetch_from_alpaca(self, ticker, start_date, end_date):
        """
        Fetch data from Alpaca API
        """
        if not self.stock_client:
            raise ValueError("Alpaca API credentials not provided")
        
        print(f"Fetching {ticker} data from Alpaca...")
        
        try:
            # Convert string dates to datetime
            start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
            
            # Request for data
            request_params = StockBarsRequest(
                symbol_or_symbols=[ticker],
                timeframe=TimeFrame.Day,
                start=start,
                end=end
            )
            
            bars = self.stock_client.get_stock_bars(request_params)
            df = bars.df.reset_index()
            
            # Check if we got any data
            if len(df) == 0:
                raise Exception(f"No data returned for {ticker} from Alpaca")
            
            # Process data for FinRL format
            df = df.rename(columns={
                'timestamp': 'date',
                'symbol': 'tic'
            })
            
            # Convert timestamp to date format
            df['date'] = pd.to_datetime(df['date']).dt.date
            
            return df
            
        except Exception as e:
            print(f"Error fetching {ticker} from Alpaca: {e}")
            return None
    
    def fetch_from_yfinance(self, ticker, start_date, end_date):
        """
        Fetch data from Yahoo Finance
        """
        print(f"Fetching {ticker} data from Yahoo Finance...")
        
        try:
            # For S&P 500 index, use ^GSPC
            symbol = ticker
            if ticker == "SPX":
                symbol = "^GSPC"
                
            df = yf.download(symbol, start=start_date, end=end_date)
            df = df.reset_index()
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Adj Close': 'adj_close'
            })
            
            # Add ticker column
            df['tic'] = ticker
            
            # Convert timestamp to date format
            df['date'] = pd.to_datetime(df['date']).dt.date
            
            return df
            
        except Exception as e:
            print(f"Error fetching {ticker} from Yahoo Finance: {e}")
            return None
    
    def fetch_data(self, ticker, start_date, end_date, prefer_source="alpaca"):
        """
        Fetch data with fallback mechanism
        """
        df = None
        
        if prefer_source == "alpaca" and self.stock_client:
            df = self.fetch_from_alpaca(ticker, start_date, end_date)
        
        # Fall back to Yahoo Finance if Alpaca fails or is not available
        if df is None or len(df) < 100:  # Not enough data
            df = self.fetch_from_yfinance(ticker, start_date, end_date)
        
        # Check if we got any data
        if df is None or len(df) < 100:
            raise Exception(f"Failed to fetch sufficient data for {ticker}")
        
        return df
    
    def fetch_benchmark_data(self, benchmark_tickers, start_date, end_date):
        """
        Fetch benchmark data from Yahoo Finance
        """
        benchmarks = {}
        
        for name, ticker in benchmark_tickers.items():
            try:
                df = yf.download(ticker, start=start_date, end=end_date)
                df['daily_return'] = df['Close'].pct_change(1)
                # Add cumulative return calculation
                df['cumulative_return'] = (1 + df['daily_return'].fillna(0)).cumprod() - 1
                benchmarks[name] = df
                print(f"Successfully fetched {name} benchmark data with {len(df)} rows")
            except Exception as e:
                print(f"Error fetching benchmark {name} ({ticker}): {e}")
        
        return benchmarks