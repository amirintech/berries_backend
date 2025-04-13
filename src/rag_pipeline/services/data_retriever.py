"""
Service to retrieve data from various sources based on processed query requirements.
"""

import json
from typing import Dict, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from .market_data import AlpacaClient, MarketDataClient
from .sec_embeddings import SECEmbeddingsManager


class DataRetrievalError(Exception):
    """Custom exception for errors during data retrieval."""
    pass


class DataRetriever:
    """
    Orchestrates data fetching from different services based on query analysis.
    Uses an LLM to refine search queries for SEC filings.
    """

    def __init__(
        self,
        alpaca_client: AlpacaClient,
        market_data_client: MarketDataClient,
        sec_manager: SECEmbeddingsManager,
        query_llm: ChatGoogleGenerativeAI
    ):
        """
        Initialize the DataRetriever with necessary clients/managers and an LLM.

        Args:
            alpaca_client: Instance of AlpacaClient.
            market_data_client: Instance of MarketDataClient.
            sec_manager: Instance of SECEmbeddingsManager.
            query_llm: An initialized LLM instance (e.g., from QueryProcessor)
                       for generating embedding search queries.
        """
        self.alpaca_client = alpaca_client
        self.market_data_client = market_data_client
        self.sec_manager = sec_manager
        self.query_llm = query_llm

        # Setup chain for generating embedding queries
        embedding_query_template = """You are an expert financial analyst assisting in searching SEC filings.
        Given the user's query and conversation history, generate a concise, keyword-focused search query suitable for a vector database search within an SEC filing.
        Focus on extracting key entities, financial terms, topics, and years mentioned or implied.
        Do NOT include conversational phrases. Output ONLY the search query string.

        Conversation History:
        {history_string}

        User's Original Query: {user_query}

        Generated Search Query:"""
        embedding_query_prompt = ChatPromptTemplate.from_template(embedding_query_template)
        self.embedding_query_chain = embedding_query_prompt | self.query_llm | StrOutputParser()


    def _generate_embedding_query(self, user_query: str, history: List[str]) -> str:
        """Generates a focused query for vector store search using an LLM."""
        print(f"🧠 Generating embedding query for: '{user_query}'")
        history_string = "\n".join(history) if history else "No history provided."
        try:
            generated_query = self.embedding_query_chain.invoke({
                "history_string": history_string,
                "user_query": user_query
            })
            # Basic cleaning
            generated_query = generated_query.strip().strip('"').strip("'")
            print(f"✅ Generated Embedding Query: '{generated_query}'")
            return generated_query
        except Exception as e:
            print(f"⚠️ Error generating embedding query: {e}. Falling back to original query.")
            return user_query


    def _format_data_for_llm(self, data_dict: Dict | List, source_name: str) -> str:
        if not data_dict:
            return f"No data found for {source_name}."
        try:
            return f"{source_name}:\n```json\n{json.dumps(data_dict, indent=2)}\n```"
        except Exception as e:
            return f"{source_name}:\n{str(data_dict)}"


    def _fetch_sec_filings(self, filings_list: List[Dict], user_query: str, history: List[str]) -> List[str]:
        """Fetch relevant snippets from required SEC filings using a generated query."""
        sec_context = []
        if not filings_list:
            return sec_context

        # Generate the focused embedding query
        embedding_query = self._generate_embedding_query(user_query, history)

        for filing_req in filings_list:
            ticker = filing_req.get("ticker")
            year = str(filing_req.get("year"))
            filing_type = filing_req.get("type")

            if not ticker or not year or not filing_type:
                sec_context.append(f"Skipping invalid SEC filing request: {filing_req}")
                continue

            try:
                print(f"🔍 Querying {ticker} {year} {filing_type} filing using query: '{embedding_query}'...")
                results = self.sec_manager.query_filing(
                    ticker=ticker,
                    year=year,
                    query=embedding_query,
                    n_results=3
                )
                if results:
                    snippets = [f"Excerpt from {ticker} {year} {filing_type} regarding '{embedding_query}': {doc.page_content}" for doc in results]
                    sec_context.append("\n".join(snippets))
                else:
                    sec_context.append(f"No relevant excerpts found in {ticker} {year} {filing_type} for the query '{embedding_query}'.")

            except Exception as e:
                print(f"⚠️ Error fetching/querying SEC filing {ticker} {year} {filing_type}: {e}")
                sec_context.append(f"Error accessing SEC filing {ticker} {year} {filing_type}.")
        return sec_context


    def _fetch_stock_prices(self, prices_list: List[Dict]) -> List[str]:
        price_context = []
        if not prices_list: return price_context
        for price_req in prices_list:
            ticker = price_req.get("ticker")
            date = price_req.get("date", "latest")
            if not ticker:
                price_context.append(f"Skipping invalid stock price request: {price_req}")
                continue
            try:
                if date.lower() == "latest":
                    print(f"📈 Fetching latest stock price for {ticker}...")
                    data = self.market_data_client.get_stock_price(ticker)
                    price_context.append(self._format_data_for_llm(data, f"Latest Stock Price for {ticker}"))
                else:
                    print(f"⏳ Fetching historical stock price for {ticker} around {date}...")
                    if len(date) == 10 and '-' in date:
                         data = self.market_data_client.get_historical_data(ticker, period="5d", end_date=date)
                    elif len(date) == 4 and date.isdigit():
                         data = self.market_data_client.get_historical_data(ticker, period="1y", end_date=f"{date}-12-31")
                    else: raise ValueError(f"Unsupported date format: {date}")
                    price_context.append(self._format_data_for_llm(data, f"Historical Stock Data for {ticker} ({date})"))
            except Exception as e:
                print(f"⚠️ Error fetching stock price for {ticker} (Date: {date}): {e}")
                price_context.append(f"Error fetching stock data for {ticker} (Date: {date}).")
        return price_context


    def retrieve_data(self, processed_query_data: Dict, user_query: str, history: List[str]) -> str:
        """
        Fetches data required by the processed query and formats it into a context string.

        Args:
            processed_query_data: The dictionary output from QueryProcessor.process_query.
            user_query: The original user query string.
            history: The conversation history (list of strings).

        Returns:
            A string containing the aggregated data formatted for LLM context.

        Raises:
            DataRetrievalError: If critical errors occur during fetching.
        """
        context_parts = []
        try:
            # Fetch Account Info
            if processed_query_data.get("requires_account_info", False):
                print("💰 Fetching account information...")
                try:
                    account_info = self.alpaca_client.get_user_account_info()
                    context_parts.append(self._format_data_for_llm(account_info, "Account Information"))
                except Exception as e:
                    print(f"⚠️ Error fetching account info: {e}")
                    context_parts.append("Error fetching account information.")

            # Fetch Positions
            if processed_query_data.get("requires_positions", False):
                print("📊 Fetching portfolio positions...")
                try:
                    positions = self.alpaca_client.get_user_positions()
                    context_parts.append(self._format_data_for_llm(positions, "Portfolio Positions"))
                except Exception as e:
                    print(f"⚠️ Error fetching positions: {e}")
                    context_parts.append("Error fetching portfolio positions.")

            # Fetch Stock Prices
            stock_price_reqs = processed_query_data.get("requires_stock_price", [])
            stock_price_context = self._fetch_stock_prices(stock_price_reqs)
            context_parts.extend(stock_price_context)

            # Fetch SEC Filings Context (passing history now)
            sec_filing_reqs = processed_query_data.get("requires_sec_filing", [])
            sec_filing_context = self._fetch_sec_filings(sec_filing_reqs, user_query, history) # Pass history
            context_parts.extend(sec_filing_context)

            # Combine all parts
            final_context = "\n\n".join(filter(None, context_parts))
            if not final_context:
                 return "No specific data context was required or fetched for this query."
            return final_context

        except Exception as e:
            print(f"❌ Unexpected error during data retrieval: {e}")
            raise DataRetrievalError(f"An unexpected error occurred during data retrieval: {e}")