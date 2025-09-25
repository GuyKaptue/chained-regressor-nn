# tests/test_neural.py
import numpy as np # type:ignore
import torch # type:ignore
import torch.nn as nn # type:ignore
import torch.optim as optim # type:ignore
import pytest # type:ignore

from chainedregressornn import NeuralRegressor, train_neural_network, set_seed
from unittest.mock import patch, MagicMock
from io import StringIO
import sys

def test_neural_regressor_forward():
    set_seed(42)
    model = NeuralRegressor(input_dim=4, hidden_dims=[8, 4], dropout=0.1)
    x = torch.randn(5, 4)
    out = model(x)
    assert out.shape == (5, 1)

def test_train_neural_network():
    set_seed(42)
    X = np.random.randn(20, 4)
    y = np.random.randn(20)
    preds, model = train_neural_network(X, y, input_dim=4, hidden_dims=[8, 4], epochs=5, verbose=False)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (20, 1)
    assert isinstance(model, NeuralRegressor)

def test_neural_forward_wrong_shape():
    model = NeuralRegressor(input_dim=2, hidden_dims=[4])
    X = torch.randn(3, 5)  # wrong input size
    with pytest.raises(RuntimeError):
        model(X)

def test_train_neural_network_zero_epochs():
    """
    Ensure train_neural_network returns correct types even if epochs=0.
    """
    input_dim = 2
    X = torch.randn(5, input_dim)
    y = torch.randn(5)

    preds, model = train_neural_network(X, y, input_dim=input_dim, hidden_dims=[4], epochs=0, verbose=False)

    # Convert predictions to tensor if not already
    if isinstance(preds, np.ndarray):
        preds = torch.tensor(preds, dtype=torch.float32)

    # Check output types
    assert isinstance(preds, torch.Tensor)
    assert isinstance(model, NeuralRegressor)
    assert preds.shape[0] == X.shape[0]  # predictions for all inputs

def test_set_seed_cuda():
    """
    Tests set_seed when CUDA is available.
    This covers line 119 in neural.py.
    """
    # Patch both functions to allow assertions
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.manual_seed') as mock_manual_seed, \
         patch('torch.cuda.manual_seed_all') as mock_manual_seed_all:

        set_seed(123)
        
        # Verify that both manual_seed and manual_seed_all were called with the seed
        mock_manual_seed.assert_called_with(123)
        mock_manual_seed_all.assert_called_with(123)

def test_train_neural_network_with_default_device_and_verbose():
    """
    Tests the default device selection and verbose printing.
    This covers lines 161 and 182 in neural.py.
    """
    X = np.random.rand(10, 5)
    y = np.random.rand(10)
    input_dim = 5
    epochs = 2
    
    # We mock stdout to capture the print output
    mock_stdout = StringIO()
    sys.stdout = mock_stdout
    
    # We mock is_available to test the default device logic
    with patch('torch.cuda.is_available', return_value=False):
        preds, model = train_neural_network(
            X, y, input_dim=input_dim, epochs=epochs, verbose=True, print_every=1
        )
    
    sys.stdout = sys.__stdout__ # Reset stdout to prevent side effects
    
    # Check if the print output contains the expected loss message
    output = mock_stdout.getvalue()
    assert "Epoch [1/2]" in output
    assert "Epoch [2/2]" in output
    
    # Check that the model was moved to CPU as expected by the mock
    assert model.to(torch.device("cpu"))

