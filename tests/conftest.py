# tests/conftest.py
import pytest # type:ignore
import pandas as pd # type:ignore
import numpy as np # type:ignore

import warnings
from sklearn.exceptions import ConvergenceWarning # type: ignore



warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names"
)


warnings.filterwarnings(
    "ignore",
    message=".*CatBoostRegressor.*__sklearn_tags__.*"
)

@pytest.fixture
def generate_sample_orders():
    """Synthetic orders DataFrame matching the given schema."""
    rng = np.random.default_rng(42)
    n = 100

    order_ids = [f"ORD-{1000+i}" for i in range(n)]
    order_dates = pd.date_range("2024-01-01", periods=n, freq="D")
    ship_dates = order_dates + pd.to_timedelta(rng.integers(1, 5, size=n), unit="D")
    ship_modes = rng.choice(["First Class", "Second Class", "Standard Class", "Same Day"], size=n)
    customer_ids = [f"CUST-{2000+i}" for i in range(n)]
    customer_names = rng.choice(["Alice Smith", "Bob Johnson", "Charlie Lee", "Dana White"], size=n)
    segments = rng.choice(["Consumer", "Corporate", "Home Office"], size=n)
    country_regions = rng.choice(["United States", "Canada", "Germany"], size=n)
    cities = rng.choice(["New York", "Toronto", "Berlin", "Munich"], size=n)
    regions = rng.choice(["East", "West", "Central", "South"], size=n)
    product_ids = [f"PROD-{3000+i}" for i in range(n)]
    categories = rng.choice(["Furniture", "Office Supplies", "Technology"], size=n)
    sub_categories = rng.choice(["Chairs", "Phones", "Binders", "Tables"], size=n)
    product_names = rng.choice(
        ["Ergonomic Chair", "Wireless Phone", "Binder Pack", "Standing Desk"], size=n
    )
    sales = np.round(rng.uniform(20, 500, size=n), 2)
    quantities = rng.integers(1, 10, size=n)
    discounts = np.round(rng.uniform(0, 0.3, size=n), 2)
    profits = np.round(sales * (0.2 + rng.uniform(-0.1, 0.1, size=n)), 2)

    return pd.DataFrame({
        "row_id": range(1, n+1),
        "order_id": order_ids,
        "order_date": order_dates,
        "ship_date": ship_dates,
        "ship_mode": ship_modes,
        "customer_id": customer_ids,
        "customer_name": customer_names,
        "segment": segments,
        "country_region": country_regions,
        "city": cities,
        "region": regions,
        "product_id": product_ids,
        "category": categories,
        "sub-category": sub_categories,
        "product_name": product_names,
        "sales": sales,
        "quantity": quantities,
        "discount": discounts,
        "profit": profits
    })