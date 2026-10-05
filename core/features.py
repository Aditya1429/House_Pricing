"""Model feature frame construction shared by training and inference."""

from __future__ import annotations

import pandas as pd

from core.config import CITIES, FURNISHING, LOCALITIES, PROPERTY_TYPES
from core.schema import PropertyInput

CATEGORICAL = ["City", "Locality", "Property_Type", "Furnishing"]
NUMERIC = [
    "Area_sqft",
    "BHK",
    "Bathrooms",
    "Floor",
    "Total_Floors",
    "Parking",
    "Property_Age",
    "Has_Gym",
    "Has_Pool",
    "Has_Lift",
    "Metro_Distance_km",
]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "Price_Lakhs"

_CATEGORIES = {
    "City": CITIES,
    "Locality": list(LOCALITIES),
    "Property_Type": PROPERTY_TYPES,
    "Furnishing": FURNISHING,
}

# Friendly names for explanations shown to users
FEATURE_LABELS = {
    "City": "City",
    "Locality": "Locality",
    "Property_Type": "Property type",
    "Furnishing": "Furnishing",
    "Area_sqft": "Area",
    "BHK": "Bedrooms",
    "Bathrooms": "Bathrooms",
    "Floor": "Floor",
    "Total_Floors": "Building height",
    "Parking": "Parking",
    "Property_Age": "Property age",
    "Has_Gym": "Gym",
    "Has_Pool": "Swimming pool",
    "Has_Lift": "Lift",
    "Metro_Distance_km": "Metro distance",
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model feature frame with fixed categorical levels."""
    X = df[FEATURES].copy()
    for col, cats in _CATEGORIES.items():
        X[col] = pd.Categorical(X[col], categories=cats)
    for col in ("Has_Gym", "Has_Pool", "Has_Lift"):
        X[col] = X[col].astype(int)
    return X


def input_to_row(p: PropertyInput) -> dict:
    return {
        "City": p.city,
        "Locality": p.locality,
        "Property_Type": p.property_type,
        "Furnishing": p.furnishing,
        "Area_sqft": p.area_sqft,
        "BHK": p.bhk,
        "Bathrooms": p.bathrooms,
        "Floor": p.floor,
        "Total_Floors": p.total_floors,
        "Parking": p.parking,
        "Property_Age": p.property_age,
        "Has_Gym": int(p.has_gym),
        "Has_Pool": int(p.has_pool),
        "Has_Lift": int(p.has_lift),
        "Metro_Distance_km": p.metro_distance_km,
    }


def input_to_frame(p: PropertyInput) -> pd.DataFrame:
    return prepare(pd.DataFrame([input_to_row(p)]))
