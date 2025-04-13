"""Main entry point for the Berries Financial Assistant."""

from berries_backend.services import AlpacaClient, MarketDataClient, SECEmbeddingsManager
from berries_backend.config import get_api_keys, PAPER_TRADING


def test_sec_embeddings():
    """Test SEC filings embeddings functionality."""
    print("\n📄 Testing SEC Filings Embeddings...")
    
    try:
        # Initialize SEC embeddings manager
        sec_manager = SECEmbeddingsManager()
        print("✅ Successfully initialized SEC embeddings manager with FinLang model")
        
        # Test with a sample filing
        ticker = "TSLA"
        year = "2023"
        
        # Get or create embeddings
        print(f"\nProcessing {ticker} {year} filing...")
        vectordb = sec_manager.get_or_create_embeddings(ticker, year)
        
        if vectordb:
            # Test querying
            print("\n🔍 Testing filing queries...")
            test_queries = [
                "What are the main risk factors related to supply chain?",
                "Describe the company's revenue growth and key drivers",
                "What are the major competitors and competitive advantages?",
                "Discuss research and development expenses and innovation strategy",
                "Explain the company's international operations and risks"
            ]
            
            for query in test_queries:
                print(f"\nQuery: {query}")
                results = sec_manager.query_filing(ticker, year, query)
                print("\nRelevant excerpts:")
                for i, doc in enumerate(results, 1):
                    print(f"\n{i}. {doc.page_content[:300]}...")
            
            # Test deletion (optional)
            # print("\nTesting vector DB deletion...")
            # deleted = sec_manager.delete_embeddings(ticker, year)
            # print(f"Vector DB deleted: {deleted}")
        
        print("\n✅ SEC embeddings test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during SEC embeddings test: {e}")


def test_market_data():
    """Test the API integrations."""
    print("\n🔍 Testing API Integrations...")
    
    try:
        # Get API keys
        api_keys = get_api_keys()
        
        # Initialize clients
        alpaca = AlpacaClient(
            api_key=api_keys["ALPACA_API_KEY"],
            secret_key=api_keys["ALPACA_SECRET_KEY"],
            paper=PAPER_TRADING
        )
        market = MarketDataClient()
        print("✅ Successfully initialized clients")
        
        # Test account info
        print("\n📊 Getting account information...")
        account_info = alpaca.get_user_account_info()
        print(f"Account Status: {account_info['status']}")
        print(f"Portfolio Value: ${account_info['portfolio_value']:,.2f}")
        print(f"Buying Power: ${account_info['buying_power']:,.2f}")
        print(f"Cash Balance: ${account_info['cash']:,.2f}")
        
        # Test positions
        print("\n📈 Getting current positions...")
        positions = alpaca.get_user_positions()
        if positions:
            print("\nCurrent Positions:")
            for pos in positions:
                print(f"{pos['symbol']}: {pos['quantity']} shares @ ${pos['current_price']:.2f}")
                print(f"Market Value: ${pos['market_value']:,.2f}")
                print(f"P/L: ${pos['unrealized_pl']:,.2f} ({pos['unrealized_plpc']:.2%})")
        else:
            print("No open positions found")
        
        # Test real-time market data
        test_symbols = ["AAPL", "GOOGL", "MSFT"]
        print("\n💰 Getting real-time market data...")
        for symbol in test_symbols:
            try:
                # Get current price data
                price_data = market.get_stock_price(symbol)
                print(f"\n{symbol} Current Data:")
                print(f"Current Price: ${price_data['price']:,.2f}")
                print(f"Today's Range: ${price_data['low']:,.2f} - ${price_data['high']:,.2f}")
                print(f"Volume: {price_data['volume']:,.0f}")
                print(f"Market Cap: ${price_data['market_cap']:,.0f}")
                if price_data['pe_ratio']:
                    print(f"P/E Ratio: {price_data['pe_ratio']:.2f}")
                if price_data['dividend_yield']:
                    print(f"Dividend Yield: {price_data['dividend_yield']:.2%}")
                print(f"50-Day Avg: ${price_data['fifty_day_avg']:,.2f}")
                print(f"200-Day Avg: ${price_data['two_hundred_day_avg']:,.2f}")
                
                # Get historical data
                hist_data = market.get_historical_data(symbol, period="5d")
                print(f"\n{symbol} 5-Day History:")
                for day in hist_data['data']:
                    print(f"{day['date']}: Open ${day['open']:,.2f}, Close ${day['close']:,.2f}, Volume {day['volume']:,.0f}")
                    
            except Exception as e:
                print(f"Error getting data for {symbol}: {e}")
                
        print("\n✅ Integration tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")


def main():
    """Main entry point."""
    # test_market_data()
    test_sec_embeddings()


if __name__ == "__main__":
    main()