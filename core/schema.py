"""Shared input schema for the form, the chatbot and the LLM extractor."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from core.config import LOCALITIES

PropertyType = Literal["Apartment", "Villa", "Independent House"]
Furnishing = Literal["Unfurnished", "Semi-Furnished", "Fully Furnished"]


class PropertyInput(BaseModel):
    locality: str
    area_sqft: float = Field(ge=250, le=10000)
    bhk: int = Field(ge=1, le=6)
    bathrooms: int = Field(ge=1, le=7)
    property_type: PropertyType = "Apartment"
    floor: int = Field(default=3, ge=0, le=80)
    total_floors: int = Field(default=10, ge=1, le=80)
    parking: int = Field(default=1, ge=0, le=3)
    property_age: int = Field(default=5, ge=0, le=60)
    furnishing: Furnishing = "Semi-Furnished"
    has_gym: bool = False
    has_pool: bool = False
    has_lift: bool = True
    metro_distance_km: float | None = Field(default=None, ge=0, le=40)

    @model_validator(mode="after")
    def _check(self) -> "PropertyInput":
        if self.locality not in LOCALITIES:
            raise ValueError(f"Unknown locality: {self.locality}")
        if self.floor > self.total_floors:
            self.total_floors = self.floor
        if self.property_type != "Apartment":
            # Houses and villas: treat as ground-anchored low-rise
            self.floor = 0
            self.total_floors = min(self.total_floors, 4)
        if self.metro_distance_km is None:
            self.metro_distance_km = LOCALITIES[self.locality].metro_km
        return self

    @property
    def city(self) -> str:
        return LOCALITIES[self.locality].city


def typical_area_for_bhk(bhk: int) -> int:
    return {1: 600, 2: 1050, 3: 1500, 4: 2200, 5: 3000, 6: 3800}.get(bhk, 1200)


def typical_bhk_for_area(area: float) -> int:
    if area < 750:
        return 1
    if area < 1250:
        return 2
    if area < 1900:
        return 3
    if area < 2700:
        return 4
    return 5
