import pandas as pd
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split

class DataProcessor:
    """
    Class for processing financial data
    """
    def __init__(self, indicators=None):
        """
        Initialize with technical indicators
        """
        self.indicators = indicators or ["macd", "rsi", "cci", "dx"]
    
    def preprocess(self, df):
        """
        Add technical indicators and other features
        """
        print("Adding technical indicators and features...")
        
        # Ensure required columns exist
        required_columns = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in dataframe")
        
        # Add technical indicators
        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=self.indicators,
            use_turbulence=True,
            user_defined_feature=False
        )
        
        processed = fe.preprocess_data(df)
        
        return processed
    
    def split_data(self, df, train_start, train_end, test_start, test_end):
        """
        Split data into training and testing sets
        """
        print("Splitting data into training and testing sets...")
        
        # Use FinRL's data_split function
        df['date'] = pd.to_datetime(df['date'])
        train_data = data_split(df, train_start, train_end)
        test_data = data_split(df, test_start, test_end)
        
        print(f"Training data shape: {train_data.shape}")
        print(f"Testing data shape: {test_data.shape}")
        
        return train_data, test_data