"""Query processing for the financial assistant."""

from __future__ import annotations
import json
from typing import Dict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser


class QueryParsingError(Exception):
    """Raised when the LLM response cannot be parsed into the expected format."""
    
    def __init__(self, message: str, raw_response: str):
        self.raw_response = raw_response
        super().__init__(f"{message}: {raw_response}")


class QueryProcessor:
    """Processes natural language queries using Google's Gemini model."""

    def __init__(self, google_api_key: str):
        """
        Initialize the query processor.

        Args:
            google_api_key: Google API key for Gemini
        """
        # Initialize LLM model
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            google_api_key=google_api_key,
            temperature=0,
            top_p=1,
        )


        template = """You are a financial query analyzer. Your task is to determine what data is needed to answer a user's question.

        Return ONLY a JSON object matching this schema:
        {{
            "requires_sec_filing": [
                {{"type": "10-K", "year": 2024, "ticker": "AAPL"}},
                {{"type": "10-Q", "year": 2022, "ticker": "IBM"}}
            ],       // List of required SEC filings (ticker, type, year)
            "requires_account_info": boolean, // Need account balances?
            "requires_positions": boolean,  // Need portfolio positions?
            "requires_stock_price": [
                {{"ticker": "AAPL", "date": "latest"}},
                {{"ticker": "TSLA", "date": "2024-03-21"}}
            ] // List of required stock prices (ticker, date - use 'latest' for current)
        }}

        Guidelines:
        - For greetings/chit-chat: set requires_account_info and requires_positions to false, and requires_sec_filing/requires_stock_price to empty lists.
        - For "last quarter" SEC filings: use "10-Q" type and the most recent year.
        - For annual reports: use "10-K" type.
        - For ambiguous SEC filing years: use the most recent relevant filing year.
        - For current stock prices: use "latest" for the date in requires_stock_price.
        - For historical stock prices: use the specified date (e.g., "YYYY-MM-DD"). If only year is mentioned, use "YYYY-12-31".
        - Include ALL mentioned or clearly implied tickers in the respective lists.
        - Map company names to ticker symbols (e.g., "Apple" -> "AAPL").
        - ONLY return valid JSON, no other text.

        Context:
        {context}"""
        self.prompt = ChatPromptTemplate.from_template(template)

        # Create chain
        self.chain = self.prompt | self.llm | StrOutputParser()


    def process_query(self, user_query: str, history: List[str]) -> Dict:
        """
        Analyze user_query in the context of history and return a dictionary
        describing the data needed according to the new schema.

        Args:
            user_query: The current user question
            history: List of previous conversation messages

        Returns:
            Dictionary specifying required data sources and parameters
            following the new schema.

        Raises:
            QueryParsingError: If LLM response cannot be parsed or validated.
        """
        context = "\n".join([*history, f"Current query: {user_query}"])

        try:
            llm_response = self.chain.invoke({"context": context})
            original_llm_response = llm_response # Keep original for error reporting

            # Remove potential markdown code block formatting
            cleaned_response = llm_response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:-3].strip()
            elif cleaned_response.startswith("```"):
                 cleaned_response = cleaned_response[3:-3].strip()

            try:
                result = json.loads(cleaned_response)
            except json.JSONDecodeError:
                raise QueryParsingError(
                    "LLM response is not valid JSON",
                    original_llm_response
                )

            # Validate required top-level fields
            required_fields = {
                "requires_sec_filing",
                "requires_account_info",
                "requires_positions",
                "requires_stock_price"
            }
            if not all(field in result for field in required_fields):
                missing = required_fields - result.keys()
                raise QueryParsingError(
                    f"Missing required fields in LLM response: {missing}",
                    original_llm_response
                )

            # Validate data types and structures
            if not isinstance(result["requires_account_info"], bool):
                raise QueryParsingError("requires_account_info must be a boolean", original_llm_response)
            if not isinstance(result["requires_positions"], bool):
                raise QueryParsingError("requires_positions must be a boolean", original_llm_response)

            # Validate list structures using helper methods
            self._validate_sec_filing_list(result["requires_sec_filing"], original_llm_response)
            self._validate_stock_price_list(result["requires_stock_price"], original_llm_response)

            return result

        except Exception as e:
            # Re-raise specific parsing errors, otherwise wrap general exceptions
            if isinstance(e, QueryParsingError):
                raise
            # Use original_llm_response if available, otherwise str(e)
            raw_resp = original_llm_response if 'original_llm_response' in locals() else str(e)
            raise QueryParsingError(f"Error processing query: {str(e)}", raw_resp)
        
        
    def _validate_sec_filing_list(self, filings: List, raw_response: str):
        """Validate the structure of the requires_sec_filing list."""
        if not isinstance(filings, list):
            raise QueryParsingError("requires_sec_filing must be a list", raw_response)
        for item in filings:
            if not isinstance(item, dict):
                raise QueryParsingError("Each item in requires_sec_filing must be a dictionary", raw_response)
            if not all(k in item for k in ["type", "year", "ticker"]):
                raise QueryParsingError("Missing keys in requires_sec_filing item (expected type, year, ticker)", raw_response)
            if not isinstance(item["type"], str) or \
               not isinstance(item["year"], int) or \
               not isinstance(item["ticker"], str):
                raise QueryParsingError("Invalid data types in requires_sec_filing item", raw_response)


    def _validate_stock_price_list(self, prices: List, raw_response: str):
        """Validate the structure of the requires_stock_price list."""
        if not isinstance(prices, list):
            raise QueryParsingError("requires_stock_price must be a list", raw_response)
        for item in prices:
            if not isinstance(item, dict):
                raise QueryParsingError("Each item in requires_stock_price must be a dictionary", raw_response)
            if not all(k in item for k in ["ticker", "date"]):
                raise QueryParsingError("Missing keys in requires_stock_price item (expected ticker, date)", raw_response)
            if not isinstance(item["ticker"], str) or \
               not isinstance(item["date"], str): # Assuming date is string ('latest' or 'YYYY-MM-DD')
                raise QueryParsingError("Invalid data types in requires_stock_price item", raw_response)
