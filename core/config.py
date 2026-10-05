"""Paths and the city -> locality registry used across the app."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
KB_DIR = ROOT / "knowledge_base"

DATA_PATH = DATA_DIR / "housing_data.csv"
SUPPORT_DB_PATH = DATA_DIR / "support.db"
METRICS_PATH = MODEL_DIR / "metrics.json"
TEST_PREDICTIONS_PATH = MODEL_DIR / "test_predictions.csv"
POINT_MODEL_PATH = MODEL_DIR / "lgbm_point.txt"
P10_MODEL_PATH = MODEL_DIR / "lgbm_p10.txt"
P90_MODEL_PATH = MODEL_DIR / "lgbm_p90.txt"


@dataclass(frozen=True)
class Locality:
    name: str
    city: str
    rate_per_sqft: int  # approximate 2024-25 resale rate, INR
    lat: float
    lon: float
    metro_km: float  # typical distance to nearest metro / suburban station
    aliases: tuple[str, ...] = ()


_LOCALITIES: list[Locality] = [
    # Bangalore
    Locality("Indiranagar", "Bangalore", 18500, 12.9784, 77.6408, 0.8, ("indira nagar",)),
    Locality("Koramangala", "Bangalore", 17000, 12.9352, 77.6245, 2.5, ("kormangala",)),
    Locality("Jayanagar", "Bangalore", 15000, 12.9250, 77.5938, 1.2, ("jaya nagar",)),
    Locality("HSR Layout", "Bangalore", 12500, 12.9116, 77.6474, 2.8, ("hsr",)),
    Locality("Hebbal", "Bangalore", 11500, 13.0358, 77.5970, 3.5, ()),
    Locality("Bellandur", "Bangalore", 10000, 12.9304, 77.6784, 4.0, ()),
    Locality("Whitefield", "Bangalore", 9000, 12.9698, 77.7500, 1.5, ("white field", "itpl")),
    Locality("Marathahalli", "Bangalore", 8500, 12.9591, 77.6974, 3.0, ("marathalli",)),
    Locality("Yelahanka", "Bangalore", 7000, 13.1007, 77.5963, 6.0, ()),
    Locality("Electronic City", "Bangalore", 6000, 12.8452, 77.6602, 5.0, ("e-city", "ecity", "e city")),
    # Mumbai
    Locality("Worli", "Mumbai", 58000, 19.0176, 72.8162, 1.5, ()),
    Locality("Bandra West", "Mumbai", 52000, 19.0596, 72.8295, 1.0, ("bandra",)),
    Locality("Juhu", "Mumbai", 48000, 19.1075, 72.8263, 2.0, ()),
    Locality("Powai", "Mumbai", 26000, 19.1176, 72.9060, 2.5, ()),
    Locality("Andheri West", "Mumbai", 28000, 19.1364, 72.8296, 1.0, ("andheri",)),
    Locality("Chembur", "Mumbai", 24000, 19.0522, 72.9005, 1.2, ()),
    Locality("Goregaon", "Mumbai", 22000, 19.1663, 72.8526, 1.0, ()),
    Locality("Borivali", "Mumbai", 20000, 19.2307, 72.8567, 1.0, ()),
    Locality("Thane West", "Mumbai", 13500, 19.2183, 72.9781, 2.0, ("thane",)),
    Locality("Kharghar", "Mumbai", 10500, 19.0473, 73.0699, 1.8, ("navi mumbai",)),
    # Pune
    Locality("Koregaon Park", "Pune", 15500, 18.5362, 73.8940, 2.0, ("kp", "koregaon")),
    Locality("Kalyani Nagar", "Pune", 14000, 18.5463, 73.9033, 1.5, ()),
    Locality("Aundh", "Pune", 12500, 18.5590, 73.8077, 3.0, ()),
    Locality("Viman Nagar", "Pune", 12000, 18.5679, 73.9143, 3.0, ()),
    Locality("Baner", "Pune", 11000, 18.5590, 73.7868, 2.5, ()),
    Locality("Kharadi", "Pune", 10000, 18.5515, 73.9348, 4.0, ()),
    Locality("Wakad", "Pune", 8500, 18.5987, 73.7650, 3.5, ()),
    Locality("Hinjewadi", "Pune", 8000, 18.5913, 73.7389, 2.0, ("hinjawadi",)),
    Locality("Hadapsar", "Pune", 7500, 18.5089, 73.9260, 4.5, ()),
    Locality("Wagholi", "Pune", 6000, 18.5808, 73.9787, 8.0, ()),
]

LOCALITIES: dict[str, Locality] = {loc.name: loc for loc in _LOCALITIES}
CITIES: list[str] = ["Bangalore", "Mumbai", "Pune"]
CITY_ALIASES: dict[str, str] = {
    "bangalore": "Bangalore",
    "bengaluru": "Bangalore",
    "blr": "Bangalore",
    "mumbai": "Mumbai",
    "bombay": "Mumbai",
    "pune": "Pune",
    "poona": "Pune",
}

PROPERTY_TYPES = ["Apartment", "Villa", "Independent House"]
FURNISHING = ["Unfurnished", "Semi-Furnished", "Fully Furnished"]


def localities_for(city: str) -> list[str]:
    return [loc.name for loc in _LOCALITIES if loc.city == city]
