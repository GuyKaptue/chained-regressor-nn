# test/__init__.py
"""
Test suite for chainedregressornn.
Provides shared utilities for test modules.
"""

from .conftest import generate_sample_orders  # example fixture import

__all__ = ["generate_sample_orders"]