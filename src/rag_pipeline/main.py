"""Main entry point for the Berries Financial Assistant."""

import traceback  # For better error printing

from rag_pipeline.services import (
    AlpacaClient, MarketDataClient, SECEmbeddingsManager,
    QueryProcessor, QueryParsingError,
    DataRetriever, DataRetrievalError,
    ResponseGenerator
)
from rag_pipeline.utils.memory import ConversationMemory
from rag_pipeline.config import (
    get_api_keys, PAPER_TRADING, DEFAULT_MEMORY_FILE,
    MAX_CONVERSATION_HISTORY
)


def test_full_pipeline():
    """Test the full RAG pipeline from query to final response with memory."""
    print("\n🚀 Testing Full RAG Pipeline with Memory...")

    try:
        # Initialization
        api_keys = get_api_keys()
        print("🔑 API keys loaded.")

        alpaca = AlpacaClient(
            api_key=api_keys["ALPACA_API_KEY"],
            secret_key=api_keys["ALPACA_SECRET_KEY"],
            paper=PAPER_TRADING
        )
        market = MarketDataClient()
        sec_manager = SECEmbeddingsManager()
        query_processor = QueryProcessor(google_api_key=api_keys["LLM_API_KEY"])
        # Pass the query_processor's LLM instance to the retriever
        data_retriever = DataRetriever(
            alpaca_client=alpaca,
            market_data_client=market,
            sec_manager=sec_manager,
            query_llm=query_processor.llm
        )
        response_generator = ResponseGenerator(google_api_key=api_keys["LLM_API_KEY"]) # Initialize generator

        # Initialize Memory
        memory = ConversationMemory(
            max_history=MAX_CONVERSATION_HISTORY,
            memory_file=DEFAULT_MEMORY_FILE
        )
        memory.clear()
        print("🧹 Cleared conversation history for test run.")

        print("✅ Successfully initialized all services and memory.")

        # Test Cases
        test_queries = [
            "Hello there!",
            "What is the current price of TSLA and its P/E ratio?",
            "Summarize the main risks for Microsoft from their latest 10-K.",
            "What's my current portfolio value?",
            "What about my positions in AAPL?", 
            "Compare Apple's 2023 performance with its 2022 performance based on their 10-K filings.",
            "Based on that comparison, what were the key revenue drivers in 2023?",
            "What is the dividend yield for Coca-Cola (KO)?",
            "Tell me about the revenue growth for GOOG in 2023 according to their annual report.",
            "What are the risks associated with NVIDIA according to its latest 10k?"
        ]

        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Pipeline Test {i} ---")
            print(f"👤 User Query: {query}")

            # Get history from memory
            history_list = memory.get_history_as_list()
            if history_list:
                print("\n📜 Current Conversation History (for Query Processor):")
                for line in history_list:
                    print(f"  {line}")

            final_answer = "Sorry, an internal error occurred." # Default fallback

            try:
                # Process Query
                print("\n1. Processing query...")
                processed_data = query_processor.process_query(query, history_list)
                print("✅ Processed Data Requirements:")
                print(processed_data)

                # Retrieve Data
                print("\n2. Retrieving data context...")
                context_string = data_retriever.retrieve_data(
                    processed_query_data=processed_data,
                    user_query=query,
                    history=history_list 
                )
                print("\n✅ Retrieved Context String:")
                print("--------------------- CONTEXT START ---------------------")
                print(context_string)
                print("---------------------- CONTEXT END ----------------------")

                # 3. Generate Response
                print("\n3. Generating final response...")
                final_answer = response_generator.generate_response(query, context_string)
                print("\n✅ Final Generated Response:")
                print("==================== RESPONSE START ====================")
                print(f"🤖 Assistant: {final_answer}")
                print("===================== RESPONSE END =====================")


            except QueryParsingError as e:
                print(f"\n❌ Error parsing query: {e}")
                print(f"Raw response: {e.raw_response}")
                final_answer = "Sorry, I had trouble understanding the requirements of your request."
                print(f"\n🤖 Assistant: {final_answer}")
            except DataRetrievalError as e:
                print(f"\n❌ Error retrieving data: {e}")
                final_answer = "Sorry, I encountered an error while trying to retrieve the necessary data."
                print(f"\n🤖 Assistant: {final_answer}")
            except Exception as e:
                 print(f"\n❌ Unexpected error during pipeline execution: {e}")
                 traceback.print_exc()
                 final_answer = "Sorry, an unexpected error occurred."
                 print(f"\n🤖 Assistant: {final_answer}")

            # Add interaction to memory
            memory.add_interaction(query, final_answer)


        print("\n✅ Full RAG Pipeline with Memory test completed!")

    except Exception as e:
        print(f"\n❌ Error during Pipeline test setup: {e}")
        traceback.print_exc()


def main():
    """Main entry point."""

    test_full_pipeline()


if __name__ == "__main__":
    main()