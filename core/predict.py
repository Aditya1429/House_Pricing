"""Inference: point estimate, calibrated range, explanations and comparables."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from core.config import DATA_PATH, METRICS_PATH, P10_MODEL_PATH, P90_MODEL_PATH, POINT_MODEL_PATH
from core.features import FEATURES, input_to_frame, input_to_row, prepare
from core.schema import PropertyInput

# Raw model features are grouped into user-facing drivers
DRIVER_GROUPS = {
    "Location": ["City", "Locality"],
    "Area": ["Area_sqft"],
    "Bedrooms and bathrooms": ["BHK", "Bathrooms"],
    "Property type": ["Property_Type"],
    "Property age": ["Property_Age"],
    "Floor position": ["Floor", "Total_Floors"],
    "Parking": ["Parking"],
    "Furnishing": ["Furnishing"],
    "Amenities": ["Has_Gym", "Has_Pool", "Has_Lift"],
    "Metro distance": ["Metro_Distance_km"],
}


@dataclass
class Estimate:
    price_lakhs: float
    low_lakhs: float
    high_lakhs: float
    price_per_sqft: float
    baseline_lakhs: float
    # driver name -> multiplicative effect in percent vs the average listing
    drivers: dict[str, float] = field(default_factory=dict)

    def top_drivers(self, n: int = 3) -> list[tuple[str, float]]:
        return sorted(self.drivers.items(), key=lambda kv: -abs(kv[1]))[:n]


class Predictor:
    def __init__(self) -> None:
        self.point = lgb.Booster(model_file=str(POINT_MODEL_PATH))
        self.p10 = lgb.Booster(model_file=str(P10_MODEL_PATH))
        self.p90 = lgb.Booster(model_file=str(P90_MODEL_PATH))
        self.metrics = json.loads(METRICS_PATH.read_text())
        self.margin = float(self.metrics.get("interval_log_margin", 0.0))
        self.data = pd.read_csv(DATA_PATH)

    def _predict_log(self, X: pd.DataFrame) -> np.ndarray:
        return self.point.predict(X[FEATURES])

    def estimate(self, p: PropertyInput) -> Estimate:
        X = input_to_frame(p)
        log_pt = float(self._predict_log(X)[0])
        log_lo = float(self.p10.predict(X)[0]) - self.margin
        log_hi = float(self.p90.predict(X)[0]) + self.margin
        # Quantile models are fitted separately; keep the range ordered around the point
        log_lo, log_hi = min(log_lo, log_pt), max(log_hi, log_pt)

        contrib = self.point.predict(X, pred_contrib=True)[0]
        per_feature = dict(zip(FEATURES, contrib[:-1]))
        baseline = float(np.exp(contrib[-1]))
        drivers = {
            group: float((np.exp(sum(per_feature[f] for f in feats)) - 1) * 100)
            for group, feats in DRIVER_GROUPS.items()
        }
        price = float(np.exp(log_pt))
        return Estimate(
            price_lakhs=price,
            low_lakhs=float(np.exp(log_lo)),
            high_lakhs=float(np.exp(log_hi)),
            price_per_sqft=price * 1e5 / p.area_sqft,
            baseline_lakhs=baseline,
            drivers=drivers,
        )

    def area_curve(self, p: PropertyInput, points: int = 25) -> pd.DataFrame:
        """Estimated price across a range of areas, holding everything else fixed."""
        areas = np.linspace(max(300, p.area_sqft * 0.5), min(9000, p.area_sqft * 1.8), points)
        row = input_to_row(p)
        frame = prepare(pd.DataFrame([{**row, "Area_sqft": a} for a in areas]))
        return pd.DataFrame({"Area_sqft": areas, "Price_Lakhs": np.exp(self._predict_log(frame))})

    def comparables(self, p: PropertyInput, n: int = 5) -> pd.DataFrame:
        df = self.data[self.data["City"] == p.city].copy()
        score = (
            (df["Locality"] != p.locality) * 1.5
            + np.abs(np.log(df["Area_sqft"] / p.area_sqft)) * 3
            + np.abs(df["BHK"] - p.bhk) * 0.6
            + np.abs(df["Property_Age"] - p.property_age) / 10
            + (df["Property_Type"] != p.property_type) * 1.0
            + (df["Furnishing"] != p.furnishing) * 0.2
        )
        df["Similarity"] = np.round(100 / (1 + score), 0)
        df["Price_per_sqft"] = np.round(df["Price_Lakhs"] * 1e5 / df["Area_sqft"], 0)
        cols = [
            "Listing_ID",
            "Locality",
            "Property_Type",
            "BHK",
            "Area_sqft",
            "Property_Age",
            "Furnishing",
            "Price_Lakhs",
            "Price_per_sqft",
            "Similarity",
        ]
        return df.sort_values("Similarity", ascending=False)[cols].head(n).reset_index(drop=True)
