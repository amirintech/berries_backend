import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from .model import StockPredictionModel # Relative import
from .data import prepare_data # Relative import


def predict_new_data(
    model: StockPredictionModel,
    new_features: np.ndarray,
    sequence_length: int,
    feature_scaler: MinMaxScaler, # Use the scaler fitted on training data
    target_scaler: MinMaxScaler, # Use the scaler fitted on training data
    device: torch.device = None
):
    """
    Makes predictions on new, unseen feature data using a trained model.

    Args:
        model (StockPredictionModel): The trained PyTorch model.
        new_features (np.ndarray): The raw feature data for prediction (n_samples, n_features).
        sequence_length (int): The sequence length the model was trained with.
        feature_scaler (MinMaxScaler): The scaler used to scale features during training.
        target_scaler (MinMaxScaler): The scaler used to scale the target during training.
        device (torch.device, optional): The device to run prediction on. Defaults to None (auto-detect GPU/CPU).

    Returns:
        np.ndarray: The predicted stock prices, inverse-transformed to original scale.
                    Returns None if prediction is not possible.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    # 1. Scale the new features using the *fitted* feature_scaler
    try:
        scaled_new_features = feature_scaler.transform(new_features)
    except ValueError as e:
         print(f"Error scaling new features: {e}. Ensure new_features has the same number of features as training data.")
         return None
    except Exception as e:
        print(f"An unexpected error occurred during feature scaling: {e}")
        return None


    # 2. Prepare the scaled features into sequences
    # We need at least sequence_length samples to make one prediction.
    # If we only have exactly sequence_length samples, prepare_data won't work as intended.
    # We need a slightly different approach for prediction on the *last* sequence.

    if len(scaled_new_features) < sequence_length:
        print(f"Need at least {sequence_length} data points for prediction, but got {len(scaled_new_features)}.")
        return None

    # Take the last 'sequence_length' points to predict the next point
    last_sequence = scaled_new_features[-sequence_length:]
    X_pred = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0) # Add batch dimension


    # 3. Make prediction
    with torch.no_grad():
        X_pred = X_pred.to(device)
        scaled_prediction = model(X_pred) # Output shape (1, 1)


    # 4. Inverse transform the prediction using the *fitted* target_scaler
    # Ensure the prediction is on CPU and is a NumPy array before inverse transform
    scaled_prediction_np = scaled_prediction.cpu().numpy()

    try:
         # Inverse transform requires a 2D array
        original_scale_prediction = target_scaler.inverse_transform(scaled_prediction_np)
    except ValueError as e:
        print(f"Error inverse transforming prediction: {e}. Check target scaler compatibility.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during inverse transformation: {e}")
        return None

    # Return the single predicted value
    return original_scale_prediction[0, 0]


def predict_sequence(
    model: StockPredictionModel,
    feature_sequence: np.ndarray, # Should be shape (sequence_length, n_features)
    feature_scaler: MinMaxScaler,
    target_scaler: MinMaxScaler,
    device: torch.device = None
):
    """Predicts the next value for a single sequence of features."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    if feature_sequence.shape[0] != model.fc1.in_features // (64 * 2): # Infer sequence length from model
         seq_len = model.fc1.in_features // (64 * 2)
         print(f"Input sequence length {feature_sequence.shape[0]} does not match model expected length {seq_len}")
         return None


    # Scale the sequence
    try:
        scaled_sequence = feature_scaler.transform(feature_sequence)
    except Exception as e:
        print(f"Error scaling feature sequence: {e}")
        return None

    X_pred = torch.tensor(scaled_sequence, dtype=torch.float32).unsqueeze(0).to(device) # Add batch dim

    # Predict
    with torch.no_grad():
        scaled_prediction = model(X_pred)

    # Inverse scale
    try:
        original_prediction = target_scaler.inverse_transform(scaled_prediction.cpu().numpy())
        return original_prediction[0, 0] # Return single value
    except Exception as e:
        print(f"Error inverse scaling prediction: {e}")
        return None
