import torch
import torch.nn as nn
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from tqdm import tqdm
import copy
import joblib

class MLP(nn.Module):
    def __init__(self, input_size=3, hidden_sizes=[64, 32, 16], output_size=1):
        super(MLP, self).__init__()

        layers = []
        prev_size = input_size

        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size, bias=True))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class ModelManager:
    def __init__(self, model, device=None, optimizer=None, criterion=None, lr=1e-3):

        # Set device automatically if not provided
        if device is None:
            print("No device specified. Automatically selecting device...")
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Using device: {self.device}")
        else:
            self.device = device
            print(f"Using specified device: {self.device}")
        # Move model to device
        self.model = model.to(self.device)
        # Default optimizer
        if optimizer is None:
            print("No optimizer provided. Using Adam with default learning rate = 1e-3.")
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer
        # Default loss function
        if criterion is None:
            print("No loss function provided. Using Mean Squared Error Loss.")
            self.criterion = nn.MSELoss()
        else:
            self.criterion = criterion
        # Store training history
        self.history = {'train_loss': [], 'val_loss': []}
        self.scaler_X = None
        self.scaler_y = None

    def load_model_weights(self, path):
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            print(f"Model weights loaded successfully from {path}")
        except Exception as e:
            print(f"Error loading model weights from {path}: {e}")

    def train(self, loader):
        self.model.train()
        total_loss = 0

        for X_batch, y_batch in tqdm(loader, desc="Training", leave=False):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            predictions = self.model(X_batch)
            loss = self.criterion(predictions, y_batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
        return total_loss / len(loader.dataset)

    def validate(self, loader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for X_batch, y_batch in tqdm(loader, desc="Validation", leave=False):
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)

                total_loss += loss.item() * X_batch.size(0)
        avg_loss = total_loss / len(loader.dataset)
        return {"loss": avg_loss}
    
    def start_training(self, train_loader, val_loader, epochs=100, patience=20, save_path="best_model.pth", verbose=True):
        """
        inputs       : train_loader, val_loader, epochs, patience, save_path, verbose
        train_loader : DataLoader for training
        val_loader   : DataLoader for validation
        epochs       : maximum number of epochs
        patience     : early stopping patience
        save_path    : path to save best model
        verbose      : print logs

        returns      : training history dictionary (optional for plotting)
        """
        best_val_loss = float("inf")
        best_model_wts = copy.deepcopy(self.model.state_dict())
        early_stop_counter = 0
        start_time = time.time()

        for epoch in range(1, epochs + 1):

            # Training
            train_loss = self.train(train_loader)

            # Validation
            val_metrics = self.validate(val_loader)
            val_loss = val_metrics["loss"]

            # Storing History
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            # self.history["val_mae"].append(val_metrics["mae"])
            # self.history["val_rmse"].append(val_metrics["rmse"])
            # self.history["val_r2"].append(val_metrics["r2"])

            # Checking for best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                torch.save(best_model_wts, save_path)
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            # Logging 
            if verbose:
                print(
                    f"Epoch [{epoch}/{epochs}] | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    # f"MAE: {val_metrics['mae']:.4f} | "
                    # f"RMSE: {val_metrics['rmse']:.4f} | "
                    # f"R2: {val_metrics['r2']:.4f}"
                )

            # Early Stopping check
            if early_stop_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping after {epoch} epochs.")
                break

        # Loading best model after training loop finishes
        self.model.load_state_dict(best_model_wts)

        total_time = time.time() - start_time

        if verbose:
            print(f"\nTraining completed in {total_time:.2f} seconds.")
            print(f"Best Validation Loss: {best_val_loss:.6f}")
        return self.history

    def load_scalers(self, path):
        scalers = joblib.load(path)
        self.scaler_X = scalers["scaler_X"]
        self.scaler_y = scalers["scaler_y"]
    
    def predict(self, x):
        """
        Run inference on input x.

        Accepts:
            - np.ndarray
            - torch.Tensor
            - Python list

        Applies scaler_X before inference and inverse-transforms
        the output with scaler_y if scalers are loaded.

        Returns:
            np.ndarray of shape (n_samples,)
        """
        # Normalise input to numpy float32
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        elif isinstance(x, list):
            x = np.array(x)

        x = x.astype(np.float32)

        # Ensure 2-D: (n_samples, n_features)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Scale features if a scaler is available
        if self.scaler_X is not None:
            x = self.scaler_X.transform(x)

        # Inference
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            preds = self.model(x_tensor)          # (n_samples, 1)  or  (n_samples,)

        preds = preds.cpu().numpy()

        # Ensure shape (n_samples, 1) for inverse_transform
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)

        # Inverse-scale predictions if a scaler is available
        if self.scaler_y is not None:
            preds = self.scaler_y.inverse_transform(preds)

        return preds.flatten()                    # return clean 1-D array