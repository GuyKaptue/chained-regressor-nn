"""
chainedregressornn
==================

A chained regression framework that combines traditional machine learning
regressors with optional deep neural network stages for enhanced predictive
performance on tabular data.

Modules
-------
core
    Contains the `ChainedRegressorNN` class — the main pipeline implementation.
neural
    Defines the `NeuralRegressor` class and training utilities for neural stages.
utils
    Provides helper functions and default regressor configurations.

Example
-------
>>> from chainedregressornn import ChainedRegressorNN
>>> model = ChainedRegressorNN(target="price")
>>> model.fit(train_df)
>>> preds = model.predict(test_df)
>>> metrics = model.evaluate(test_df)
"""

from .core import ChainedRegressorNN
from .neural import NeuralRegressor, train_neural_network, set_seed
from .utils import safe_hstack_hybrid, DEFAULT_REGRESSORS

__all__ = [
    "ChainedRegressorNN",
    "NeuralRegressor",
    "train_neural_network",
    "set_seed",
    "safe_hstack_hybrid",
    "DEFAULT_REGRESSORS",
]

__version__ = "0.1.1"
__author__ = "Guy Kaptue"
__license__ = "MIT"
