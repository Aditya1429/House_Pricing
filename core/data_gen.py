"""Reproducible synthetic housing dataset for Bangalore, Mumbai and Pune.

Prices are built from locality base rates (approximate 2024-25 resale levels)
with non-linear effects and multiplicative noise, so models have to learn
interactions rather than one linear formula. Run: python -m core.data_gen
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import DATA_DIR, DATA_PATH, LOCALITIES

ROWS_PER_CITY = 2000
SEED = 42

# Typical area bands per BHK (sq.ft); Mumbai homes are smaller for the same BHK
_AREA_BY_BHK = {1: (450, 750), 2: (800, 1250), 3: (1200, 1900), 4: (1800, 2800), 5: (2600, 4200)}
_CITY_AREA_FACTOR = {"Bangalore": 1.0, "Mumbai": 0.78, "Pune": 0.95}
_TYPE_PROBS = {
    "Bangalore": [0.80, 0.08, 0.12],
    "Mumbai": [0.95, 0.02, 0.03],
    "Pune": [0.82, 0.06, 0.12],
}
_TYPE_PREMIUM = {
    "Bangalore": {"Apartment": 1.0, "Villa": 1.22, "Independent House": 1.10},
    "Mumbai": {"Apartment": 1.0, "Villa": 1.45, "Independent House": 1.25},
    "Pune": {"Apartment": 1.0, "Villa": 1.18, "Independent House": 1.06},
}
_FURNISH_PREMIUM = {"Unfurnished": 1.0, "Semi-Furnished": 1.035, "Fully Furnished": 1.08}


def generate(rows_per_city: int = ROWS_PER_CITY, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    by_city: dict[str, list] = {}
    for loc in LOCALITIES.values():
        by_city.setdefault(loc.city, []).append(loc)

    for city, locs in by_city.items():
        # More listings in mid-market localities than in the most expensive ones
        rates = np.array([l.rate_per_sqft for l in locs], dtype=float)
        weights = 1 / np.sqrt(rates)
        weights /= weights.sum()
        for _ in range(rows_per_city):
            loc = locs[rng.choice(len(locs), p=weights)]
            ptype = rng.choice(["Apartment", "Villa", "Independent House"], p=_TYPE_PROBS[city])

            bhk = int(rng.choice([1, 2, 3, 4, 5], p=[0.14, 0.38, 0.32, 0.12, 0.04]))
            if ptype != "Apartment":
                bhk = max(bhk, 3)
            lo, hi = _AREA_BY_BHK[bhk]
            area = rng.uniform(lo, hi) * _CITY_AREA_FACTOR[city]
            if ptype == "Villa":
                area *= rng.uniform(1.3, 1.7)
            area = float(np.clip(round(area / 5) * 5, 300, 9000))

            bathrooms = int(np.clip(bhk + rng.choice([-1, 0, 0, 1]), 1, bhk + 1))

            if ptype == "Apartment":
                tower = rng.choice([4, 8, 12, 18, 25, 35, 45] if city == "Mumbai" else [4, 5, 8, 12, 14, 20, 25])
                total_floors = int(tower)
                floor = int(rng.integers(0, total_floors + 1))
                has_lift = total_floors > 4 or rng.random() < 0.5
            else:
                total_floors = int(rng.integers(1, 4))
                floor = 0
                has_lift = rng.random() < 0.15

            age = int(np.clip(rng.gamma(2.0, 4.5), 0, 40))
            parking = int(rng.choice([0, 1, 2], p=[0.25, 0.55, 0.20] if city == "Mumbai" else [0.12, 0.58, 0.30]))
            if ptype != "Apartment":
                parking = max(parking, 1)
            furnishing = rng.choice(list(_FURNISH_PREMIUM), p=[0.40, 0.42, 0.18])
            gated = ptype == "Villa" or (ptype == "Apartment" and total_floors >= 8)
            has_gym = bool(gated and rng.random() < 0.7 - age * 0.012)
            has_pool = bool(gated and rng.random() < 0.45 - age * 0.01)
            metro = float(np.clip(rng.normal(loc.metro_km, loc.metro_km * 0.4), 0.2, 25))

            # --- price construction ---
            rate = loc.rate_per_sqft * rng.lognormal(0, 0.05)  # micro-location variation
            m = _TYPE_PREMIUM[city][ptype]
            m *= 1.0 - 0.55 * (1 - np.exp(-age / 18))  # depreciation curve, flattens with age
            m *= 1.0 + 0.10 * np.exp(-metro / 1.5)  # metro proximity premium
            if ptype == "Apartment" and total_floors > 0:
                height_ratio = floor / total_floors
                floor_prem = 0.12 if city == "Mumbai" else 0.05  # views matter more in Mumbai
                m *= 1.0 + floor_prem * height_ratio - (0.04 if floor == 0 else 0)
            m *= _FURNISH_PREMIUM[furnishing]
            m *= 1.0 + 0.03 * has_gym + 0.04 * has_pool + (0.0 if has_lift or floor <= 2 else -0.06)
            # Large homes in premium areas carry a luxury premium
            if area > 2000 and loc.rate_per_sqft > 15000:
                m *= 1.0 + 0.08 * min((area - 2000) / 2000, 1.0)
            # Bathroom shortfall penalty
            if bathrooms < bhk:
                m *= 0.96

            price_inr = area * rate * m
            price_inr += parking * (300000 if city != "Mumbai" else 900000) * (loc.rate_per_sqft / 15000)
            noise_sd = 0.07 + 0.03 * (ptype != "Apartment")  # heteroskedastic noise
            price_inr *= rng.lognormal(0, noise_sd)
            price_lakhs = round(price_inr / 1e5, 2)

            records.append(
                {
                    "City": city,
                    "Locality": loc.name,
                    "Property_Type": ptype,
                    "Area_sqft": area,
                    "BHK": bhk,
                    "Bathrooms": bathrooms,
                    "Floor": floor,
                    "Total_Floors": total_floors,
                    "Parking": parking,
                    "Property_Age": age,
                    "Furnishing": furnishing,
                    "Has_Gym": has_gym,
                    "Has_Pool": has_pool,
                    "Has_Lift": bool(has_lift),
                    "Metro_Distance_km": round(metro, 2),
                    "Price_Lakhs": price_lakhs,
                }
            )

    df = pd.DataFrame(records)
    df.insert(0, "Listing_ID", [f"L{i:05d}" for i in range(1, len(df) + 1)])
    return df


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    df = generate()
    df.to_csv(DATA_PATH, index=False)
    print(f"Wrote {len(df)} rows to {DATA_PATH}")
    print(df.groupby("City")["Price_Lakhs"].describe().round(1))


if __name__ == "__main__":
    main()
