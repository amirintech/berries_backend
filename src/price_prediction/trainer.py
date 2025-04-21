import torch
from torch.utils.data import DataLoader, Dataset
import time
from .model import StockPredictionModel # Relative import

def train_model(
    model: StockPredictionModel,
    train_loader: DataLoader,
    criterion,
    optimizer,
    num_epochs: int = 30,
    device: torch.device = None
):
    """Trains the stock prediction model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Starting training on {device}...")
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}, Time: {epoch_time:.2f}s")

    print("Training finished.")
    return model # Return trained model


def evaluate_model(
    model: StockPredictionModel,
    test_loader: DataLoader,
    criterion,
    device: torch.device = None
):
    """Evaluates the model on the test set."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    total_loss = 0.0
    y_true = []
    y_pred = []

    print(f"Starting evaluation on {device}...")
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            total_loss += loss.item()

            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    avg_test_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.6f}")

    return avg_test_loss, y_true, y_pred
