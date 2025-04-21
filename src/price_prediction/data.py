import torch
from torch.utils.data import Dataset
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from stockstats import StockDataFrame
import pandas as pd
import numpy as np


class StockDataset(Dataset):
    """PyTorch Dataset for stock data."""
    def __init__(self, X, y):
        if not isinstance(X, torch.Tensor):
            raise TypeError(f"Expected X to be a torch.Tensor, but got {type(X)}")
        if not isinstance(y, torch.Tensor):
            raise TypeError(f"Expected y to be a torch.Tensor, but got {type(y)}")
        self.X = X
        self.y = y


    def __len__(self):
        return len(self.X)


    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_data(features: np.ndarray, target: np.ndarray, sequence_length: int):
    """
    Prepares data into sequences for LSTM input.

    Args:
        features (np.ndarray): Array of feature data.
        target (np.ndarray): Array of target data.
        sequence_length (int): The length of each input sequence.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors of sequences (X) and corresponding targets (y).
    """
    X, y = [], []
    if len(features) <= sequence_length:
        raise ValueError("Not enough data points to create sequences of the specified length.")

    for i in range(len(features) - sequence_length):
        X.append(features[i : i + sequence_length])
        # Ensure target shape is consistent, assuming target is (n_samples, 1) initially
        y.append(target[i + sequence_length])

    # Ensure y has the correct shape before converting to tensor, e.g., (n_sequences, 1)
    y = np.array(y).reshape(-1, 1)

    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def get_stock_data(ticker: str, start: str, end: str):
    """
    Downloads stock data, calculates technical indicators, and scales features and target.

    Args:
        ticker (str): Stock ticker symbol.
        start (str): Start date for data download (YYYY-MM-DD).
        end (str): End date for data download (YYYY-MM-DD).

    Returns:
        Tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler, pd.DataFrame]:
            Scaled features, scaled target, feature scaler, target scaler, original data with indicators.
    """
    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker} between {start} and {end}")

        stock = StockDataFrame.retype(data.copy())

        # Calculate technical indicators
        # Ensure columns exist before assignment to avoid KeyError
        indicator_cols = {
            'rsi_14': 'RSI', 'macd': 'MACD', 'macds': 'MACD_signal',
            'macdh': 'MACD_diff', 'boll_ub': 'BB_upper', 'boll_lb': 'BB_lower'
        }
        for tech_col, data_col in indicator_cols.items():
            if tech_col in stock.columns:
                data[data_col] = stock[tech_col]
            else:
                 print(f"Warning: Technical indicator '{tech_col}' not found for {ticker}.")
                 data[data_col] = np.nan # Add column with NaNs if indicator missing

        data.dropna(inplace=True)
        if data.empty:
            raise ValueError(f"Data became empty after dropping NaNs for ticker {ticker}. Check indicators.")

        target = data[['Close']].values # Keep as 2D array

        # Define features - ensure these columns exist after indicator calculation and dropna
        feature_list = ['High', 'Low', 'Open', 'Volume'] + list(indicator_cols.values())
        available_features = [f for f in feature_list if f in data.columns]
        features = data[available_features].values

        if features.shape[1] == 0:
            raise ValueError(f"No features available for ticker {ticker} after processing.")


        # Scale features and target
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()

        # Ensure target is 2D before scaling
        if target.ndim == 1:
            target = target.reshape(-1, 1)

        scaled_features = feature_scaler.fit_transform(features)
        scaled_target = target_scaler.fit_transform(target)

        return scaled_features, scaled_target, feature_scaler, target_scaler, data, available_features

    except Exception as e:
        print(f"Error processing data for {ticker}: {e}")
        # Return empty arrays/scalers and None for data to signal failure
        return np.array([]), np.array([]), MinMaxScaler(), MinMaxScaler(), None, []
