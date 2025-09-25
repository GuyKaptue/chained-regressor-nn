"""
neural.py
=========

Neural network components for the ChainedRegressorNN pipeline.

This module provides:
---------------------
1. NeuralRegressor — a deep, configurable, regularized MLP for regression tasks,
   with optional residual connections, batch normalization, and dropout.
2. train_neural_network — utility to train a NeuralRegressor on tabular data.
3. set_seed — helper to ensure reproducible results across NumPy, Python's
   `random`, and PyTorch.

Author
------
Guy Kaptue
"""
# type: ignore
import torch # type: ignore
import torch.nn as nn # type:ignore
import torch.optim as optim # type:ignore
import numpy as np # type:ignore
import random


class NeuralRegressor(nn.Module):
    """
    A deep, regularized, optionally residual-connected MLP for regression.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dims : list[int]
        Sizes of hidden layers.
    dropout : float, default=0.2
        Dropout probability.
    activation : str, default="gelu"
        Activation function: 'relu', 'gelu', or 'selu'.
    use_batchnorm : bool, default=True
        Whether to use BatchNorm1d after each linear layer.
    residual : bool, default=True
        Whether to add skip connections between every other layer.
    """

    def __init__(self, input_dim, hidden_dims, dropout=0.2, activation="gelu",
                 use_batchnorm=True, residual=True):
        super().__init__()

        act_map = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "selu": nn.SELU()
        }
        act_fn = act_map.get(activation.lower(), nn.ReLU())

        layers = []
        prev_dim = input_dim
        self.residual = residual

        for h in hidden_dims:
            block = [nn.Linear(prev_dim, h)]
            if use_batchnorm:
                block.append(nn.BatchNorm1d(h))
            block.append(act_fn)
            if dropout > 0:
                block.append(nn.Dropout(dropout))
            layers.append(nn.Sequential(*block))
            prev_dim = h

        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(prev_dim, 1)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor or numpy.ndarray
            Input feature matrix.

        Returns
        -------
        torch.Tensor
            Predicted values.
        """
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, dtype=torch.float32)
        out = x
        prev_out = None

        for layer in self.hidden_layers:
            new_out = layer(out)
            if self.residual and prev_out is not None and new_out.shape == prev_out.shape:
                out = new_out + prev_out
                prev_out = new_out
            else:
                prev_out = out
                out = new_out

        return self.output_layer(out)


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.

    Parameters
    ----------
    seed : int, default=42
        Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_neural_network(X, y, input_dim, epochs, hidden_dims=None,
                         lr=0.001, dropout=0.2, verbose=True,
                         device=None, seed=42, print_every=50):
    """
    Train a neural network for regression.

    Parameters
    ----------
    X : array-like
        Input features (NumPy array, sparse matrix, or torch.Tensor).
    y : array-like
        Target values.
    input_dim : int
        Number of input dimensions.
    hidden_dims : list[int], optional
        Hidden layer dimensions. If None, uses default architecture.
    lr : float, default=0.001
        Learning rate.
    epochs : int, default=500
        Number of training epochs.
    dropout : float, default=0.2
        Dropout rate.
    verbose : bool, default=True
        Whether to print training progress.
    device : torch.device or None, default=None
        Device to train on. If None, auto-selects CUDA if available.
    seed : int, default=42
        Random seed for reproducibility.
    print_every : int, default=50
        Print loss every N epochs.

    Returns
    -------
    tuple
        (predictions : numpy.ndarray, trained_model : NeuralRegressor)
    """
    set_seed(seed)

    if hidden_dims is None:
        hidden_dims = [512, 256, 256, 128, 64, 64, 32, 16]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.as_tensor(y.reshape(-1, 1), dtype=torch.float32, device=device)

    model = NeuralRegressor(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        if verbose and (epoch % print_every == 0 or epoch == epochs - 1):
            print(f"  Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy()

    return preds, model
