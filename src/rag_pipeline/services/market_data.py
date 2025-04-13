"""Market data and trading services."""

from typing import Dict, List
from datetime import datetime
import yfinance as yf
from alpaca.trading.client import TradingClient


class AlpacaClient:
    """Client for Alpaca trading API, handling user account and positions."""
    
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        """
        Initialize Alpaca trading client.
        
        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            paper: Whether to use paper trading (default True)
            
        Raises:
            ValueError: If API keys are missing
        """
        if not api_key or not secret_key:
            raise ValueError("Alpaca API key and secret key are required")
            
        self.trading_client = TradingClient(api_key, secret_key, paper=paper)
        
    def get_user_account_info(self) -> Dict:
        """
        Get user account information from Alpaca API.

        Returns:
            A dictionary containing account details
            
        Raises:
            Exception: If fetching account info fails
        """
        try:
            account = self.trading_client.get_account()
            return {
                "account_id": str(account.id),
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "buying_power": float(account.buying_power),
                "equity": float(account.equity),
                "last_equity": float(account.last_equity),
                "long_market_value": float(account.long_market_value),
                "short_market_value": float(account.short_market_value),
                "initial_margin": float(account.initial_margin),
                "maintenance_margin": float(account.maintenance_margin),
                "status": account.status
            }
        except Exception as e:
            raise Exception(f"Error fetching account info: {str(e)}")

    def get_user_positions(self) -> List[Dict]:
        """
        Get user portfolio positions from Alpaca API.

        Returns:
            A list of dictionaries containing position details
            
        Raises:
            Exception: If fetching positions fails
        """
        try:
            positions = self.trading_client.get_all_positions()
            return [
                {
                    "symbol": p.symbol,
                    "quantity": float(p.qty),
                    "market_value": float(p.market_value),
                    "cost_basis": float(p.cost_basis),
                    "unrealized_pl": float(p.unrealized_pl),
                    "unrealized_plpc": float(p.unrealized_plpc),
                    "current_price": float(p.current_price),
                    "lastday_price": float(p.lastday_price),
                    "change_today": float(p.change_today)
                }
                for p in positions
            ]
        except Exception as e:
            raise Exception(f"Error fetching positions: {str(e)}")


class MarketDataClient:
    """Client for real-time market data using Yahoo Finance."""
    
    def get_stock_price(self, ticker: str) -> Dict:
        """
        Get current stock price and market data.

        Args:
            ticker: The stock ticker symbol

        Returns:
            A dictionary containing price and market data
            
        Raises:
            Exception: If fetching stock price fails
        """
        ticker = ticker.upper()
        try:
            # Get stock info from Yahoo Finance
            stock = yf.Ticker(ticker)
            
            # Get real-time quote data
            info = stock.info
            
            # Get today's data
            hist = stock.history(period="1d")
            
            if hist.empty:
                raise Exception("No trading data available")
                
            result = {
                "symbol": ticker,
                "price": info.get('regularMarketPrice', 0.0),
                "time": datetime.now().isoformat(),
                "ask_price": info.get('ask', 0.0),
                "ask_size": info.get('askSize', 0),
                "bid_price": info.get('bid', 0.0),
                "bid_size": info.get('bidSize', 0),
                "open": float(hist['Open'].iloc[-1]),
                "high": float(hist['High'].iloc[-1]),
                "low": float(hist['Low'].iloc[-1]),
                "volume": float(hist['Volume'].iloc[-1]),
                # Additional data from Yahoo Finance
                "market_cap": info.get('marketCap', 0),
                "pe_ratio": info.get('trailingPE', None),
                "dividend_yield": info.get('dividendYield', None),
                "fifty_day_avg": info.get('fiftyDayAverage', None),
                "two_hundred_day_avg": info.get('twoHundredDayAverage', None)
            }
            
            return result
            
        except Exception as e:
            raise Exception(f"Error fetching stock price for {ticker}: {str(e)}")
            
    def get_historical_data(self, ticker: str, period: str = "1mo") -> Dict:
        """
        Get historical price data for a stock.
        
        Args:
            ticker: The stock ticker symbol
            period: Time period to fetch (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            Dictionary with historical price data
            
        Raises:
            Exception: If fetching historical data fails
        """
        ticker = ticker.upper()
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            if hist.empty:
                raise Exception("No historical data available")
                
            return {
                "symbol": ticker,
                "period": period,
                "data": [
                    {
                        "date": index.strftime("%Y-%m-%d"),
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "volume": float(row["Volume"])
                    }
                    for index, row in hist.iterrows()
                ]
            }
            
        except Exception as e:
            raise Exception(f"Error fetching historical data for {ticker}: {str(e)}") 