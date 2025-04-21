import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .data import get_stock_data, prepare_data, StockDataset
from .model import StockPredictionModel
from .trainer import train_model, evaluate_model
from .predictor import predict_sequence

def run_stock_prediction_test():
    """Main function to run the stock prediction workflow."""
    # --- Configuration ---
    ticker = 'NVDA' # Example: NVIDIA
    start_date = '2020-01-01'
    end_date = '2024-05-01' # Use a recent end date
    sequence_length = 20 # Using a slightly longer sequence
    test_split_ratio = 0.15
    batch_size = 32
    num_epochs = 25 # Moderate number for testing, increase for better results
    learning_rate = 0.001

    # --- Data Loading and Preparation ---
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    scaled_features, scaled_target, feature_scaler, target_scaler, raw_data, feature_names = get_stock_data(
        ticker, start_date, end_date
    )

    # Check if data loading was successful
    if raw_data is None or not isinstance(raw_data, pd.DataFrame) or raw_data.empty or scaled_features.size == 0:
        print("Failed to get valid data. Exiting.")
        return

    print(f"Data fetched successfully. Features used: {feature_names}")
    print(f"Raw data shape: {raw_data.shape}")
    print(f"Scaled features shape: {scaled_features.shape}")
    print(f"Scaled target shape: {scaled_target.shape}")

    print(f"Preparing data with sequence length {sequence_length}...")
    try:
        # Prepare data expects numpy arrays
        X, y = prepare_data(scaled_features, scaled_target, sequence_length)
        print(f"Data prepared. X shape: {X.shape}, y shape: {y.shape}")
    except ValueError as e:
        print(f"Error preparing data: {e}. Exiting.")
        return

    # Ensure we have features after preparation
    if X.shape[-1] == 0:
         print("Error: No features found in the prepared data (X). Exiting.")
         return
    n_features = X.shape[-1]

    # --- Dataset and DataLoader ---
    full_dataset = StockDataset(X, y)
    total_size = len(full_dataset)
    test_size = int(total_size * test_split_ratio)
    train_size = total_size - test_size

    if train_size <= 0 or test_size <= 0:
        print(f"Dataset too small for train/test split with ratio {test_split_ratio}. Total: {total_size}, Train: {train_size}, Test: {test_size}. Exiting.")
        return

    print(f"Splitting data: Total size = {total_size}, Train size = {train_size}, Test size = {test_size}")
    # Use indices for splitting to easily track dates later if needed
    indices = list(range(total_size))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Shuffle training data
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # No shuffle for test

    # --- Model Initialization ---
    print("Initializing model...")
    model = StockPredictionModel(sequence_length=sequence_length, n_features=n_features)
    print(model) # Print model architecture summary

    # --- Loss and Optimizer ---
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- Training ---
    print("\n--- Training ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model = train_model(model, train_loader, criterion, optimizer, num_epochs, device)

    # --- Evaluation ---
    print("\n--- Evaluation ---")
    avg_test_loss, y_true_eval_scaled, y_pred_eval_scaled = evaluate_model(trained_model, test_loader, criterion, device)

    # Inverse transform for plotting evaluation results
    try:
        y_true_eval = target_scaler.inverse_transform(np.array(y_true_eval_scaled).reshape(-1, 1))
        y_pred_eval = target_scaler.inverse_transform(np.array(y_pred_eval_scaled).reshape(-1, 1))
    except ValueError as e:
        print(f"Error during inverse transform for evaluation plot: {e}")
        # Continue without plotting if inverse transform fails
        y_true_eval, y_pred_eval = None, None


    # --- Plotting Evaluation Results ---
    if y_true_eval is not None and y_pred_eval is not None:
        plt.figure(figsize=(14, 7))
        plt.plot(y_true_eval, label="Actual Prices (Test Set)", color='blue', marker='.', linestyle='-')
        plt.plot(y_pred_eval, label="Predicted Prices (Test Set)", color='red', marker='.', linestyle='--')
        plt.title(f'{ticker} Stock Price Prediction - Test Set Evaluation (Epochs: {num_epochs})')
        plt.xlabel('Time Steps (Test Set)')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # You might want to save the plot instead of showing interactively
        # plt.savefig(f"{ticker}_test_evaluation_{num_epochs}epochs.png")
        plt.show()
    else:
        print("Skipping evaluation plot due to inverse transform error.")


    # --- Example Prediction ---
    print("\n--- Example Prediction ---")
    # Use the last sequence from the *original* scaled_features data to predict the next step
    if len(scaled_features) >= sequence_length:
        # Ensure raw_data index is DatetimeIndex for proper date handling
        if not isinstance(raw_data.index, pd.DatetimeIndex):
            try:
                raw_data.index = pd.to_datetime(raw_data.index)
            except Exception as e:
                print(f"Could not convert raw_data index to DatetimeIndex: {e}")


        # Get the last sequence of features available in the dataset
        last_feature_sequence = scaled_features[-sequence_length:] # Shape: (sequence_length, n_features)

        predicted_next_price = predict_sequence(
            model=trained_model,
            feature_sequence=last_feature_sequence,
            feature_scaler=feature_scaler, # Use the scaler fitted on training data
            target_scaler=target_scaler, # Use the scaler fitted on training data
            device=device
        )

        if predicted_next_price is not None:
             # Try to find the date corresponding to the prediction
            last_date_in_dataset = None
            if isinstance(raw_data.index, pd.DatetimeIndex) and len(raw_data) > 0:
                 # The last feature sequence ends at index -1 of raw_data
                 # The target it predicts corresponds to the *next* day's close price
                 # This assumes daily data and no gaps right at the end.
                 try:
                     # raw_data was used to generate scaled_features, indices should align
                     last_date_in_dataset = raw_data.index[-1]
                     prediction_date = last_date_in_dataset + pd.Timedelta(days=1) # Approximate date
                     print(f"Predicted price for approx. {prediction_date.date()}: {predicted_next_price:.2f}")
                 except IndexError:
                     print("Could not determine last date from raw_data index.")
                     print(f"Predicted next price: {predicted_next_price:.2f}")
                 except Exception as e:
                     print(f"Error calculating prediction date: {e}")
                     print(f"Predicted next price: {predicted_next_price:.2f}")

            else:
                 print(f"Predicted next price (date unknown): {predicted_next_price:.2f}")
        else:
            print("Could not generate example prediction.")
    else:
        print(f"Not enough data ({len(scaled_features)} points) to form a sequence of length {sequence_length} for prediction.")


if __name__ == "__main__":
    run_stock_prediction_test()