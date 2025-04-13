"""SEC filings embeddings management using ChromaDB and LangChain."""

import os
from typing import Optional, List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PDFPlumberLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from berries_backend.config import VECTOR_DB_DIR, PROJECT_ROOT


class SECEmbeddingsManager:
    """Manages embeddings of SEC filings using ChromaDB."""
    
    def __init__(self):
        """Initialize the SEC embeddings manager with financial domain embeddings."""
        self.embeddings = HuggingFaceEmbeddings(
            model_name="FinLang/finance-embeddings-investopedia",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
    def _get_vector_db_path(self, ticker: str, year: str) -> str:
        """Get the path for storing vector DB for a specific ticker and year."""
        return os.path.join(VECTOR_DB_DIR, f"{ticker.lower()}_{year}")
        
    def _get_filing_path(self, ticker: str, year: str) -> Optional[str]:
        """Get the path to the SEC filing file."""
        return os.path.join(PROJECT_ROOT, "sec_filings", "10K", ticker.upper(), f"{year}.pdf")        
        
    def get_or_create_embeddings(self, ticker: str, year: str) -> Optional[Chroma]:
        """
        Get existing vector DB or create new embeddings for SEC filing.
        
        Args:
            ticker: Stock ticker symbol
            year: Filing year
            
        Returns:
            ChromaDB instance if successful, None if filing not found
            
        Raises:
            Exception: If there's an error processing the filing
        """
        try:
            ticker = ticker.upper()
            vector_db_path = self._get_vector_db_path(ticker, year)
            
            # Check if vector DB already exists
            if os.path.exists(vector_db_path):
                return Chroma(
                    persist_directory=vector_db_path,
                    embedding_function=self.embeddings
                )
            
            # Get filing path
            filing_path = self._get_filing_path(ticker, year)
            if not filing_path:
                print(f"No filing found for {ticker} {year}")
                return None
                
            print(f"Creating new vector DB for {ticker} {year}...")
            
            # Load and split the document
            loader = PDFPlumberLoader(filing_path)
            document = loader.load()
            texts = self.text_splitter.split_documents(document)
            
            # Create and persist vector DB
            vectordb = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings,
                persist_directory=vector_db_path
            )
            vectordb.persist()
            print(f"Successfully created vector DB for {ticker} {year}")
            
            return vectordb
            
        except Exception as e:
            raise Exception(f"Error processing SEC filing for {ticker} {year}: {str(e)}")
            
    def query_filing(self, 
                    ticker: str, 
                    year: str, 
                    query: str, 
                    n_results: int = 4) -> List:
        """
        Query the SEC filing vector DB.
        
        Args:
            ticker: Stock ticker symbol
            year: Filing year
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of relevant document chunks
            
        Raises:
            Exception: If there's an error querying the filing
        """
        try:
            vectordb = self.get_or_create_embeddings(ticker, year)
            if not vectordb:
                raise Exception(f"No vector DB found for {ticker} {year}")
                
            results = vectordb.similarity_search(
                query=query,
                k=n_results
            )
            
            return results
            
        except Exception as e:
            raise Exception(f"Error querying SEC filing for {ticker} {year}: {str(e)}")
            
    def delete_embeddings(self, ticker: str, year: str) -> bool:
        """
        Delete the vector DB for a specific filing.
        
        Args:
            ticker: Stock ticker symbol
            year: Filing year
            
        Returns:
            True if deleted successfully, False if not found
        """
        try:
            vector_db_path = self._get_vector_db_path(ticker, year)
            if os.path.exists(vector_db_path):
                import shutil
                shutil.rmtree(vector_db_path)
                print(f"Deleted vector DB for {ticker} {year}")
                return True
            return False
        except Exception as e:
            raise Exception(f"Error deleting vector DB for {ticker} {year}: {str(e)}") 