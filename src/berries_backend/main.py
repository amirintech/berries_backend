"""Main entry point for the Berries Financial Assistant."""

from berries_backend.services import (
    AlpacaClient, MarketDataClient, SECEmbeddingsManager, 
    QueryProcessor, QueryParsingError
)
from berries_backend.config import get_api_keys, PAPER_TRADING




def main():
    """Main entry point."""
    # test_query_processor()  # Test query processing first


if __name__ == "__main__":
    main()