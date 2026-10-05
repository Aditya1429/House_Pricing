"""Slot-filling conversation engine for the Price Assistant (UI-independent)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from core import llm
from core.config import LOCALITIES, localities_for
from core.parser import looks_like_support_query, parse
from core.predict import Estimate, Predictor
from core.schema import PropertyInput, typical_area_for_bhk, typical_bhk_for_area

_RESET = re.compile(r"\b(start over|reset|new search|clear|forget it)\b", re.I)
_GREETING = re.compile(r"^\s*(hi|hello|hey|namaste|good (morning|afternoon|evening))\b", re.I)
_ACCURACY = re.compile(r"\b(accura|reliable|trust|how (good|correct)|error|mape|confiden)", re.I)
_HOW = re.compile(r"\b(how does (this|it) work|how do you (calculate|estimate)|what model)\b", re.I)


@dataclass
class Reply:
    text: str
    slots: dict
    estimate: Estimate | None = None
    property: PropertyInput | None = None
    assumptions: list[str] = field(default_factory=list)
    used_llm: bool = False


def _fmt_lakhs(v: float) -> str:
    return f"₹{v / 100:.2f} Cr" if v >= 100 else f"₹{v:.1f} L"


def _extract(message: str) -> tuple[dict, str | None, bool]:
    if llm.is_enabled():
        out = llm.extract_fields(message)
        if out is not None:
            fields, city = out
            # Keep only fields the rule parser can also validate
            fields = {k: v for k, v in fields.items() if k != "locality" or v in LOCALITIES}
            return fields, city, True
    r = parse(message)
    return r.fields, r.city, False


def respond(message: str, slots: dict, predictor: Predictor) -> Reply:
    if _RESET.search(message):
        return Reply("Cleared. Tell me about the property you have in mind, for example: 3 BHK in Baner, 1400 sq.ft.", {})

    fields, city, used_llm = _extract(message)

    if not fields and not city:
        if _GREETING.search(message):
            return Reply(
                "Hello. Describe a property and I will estimate its price, for example: "
                "2 BHK in Whitefield, 1100 sq.ft, 5 years old.",
                slots,
            )
        if _ACCURACY.search(message):
            m = predictor.metrics
            return Reply(
                f"On held-out test data the model's typical error is about {m['production_test_mape']:.1f}% "
                f"(MAPE), with R² of {m['production_test_r2']:.3f}. The price range I show is calibrated so that "
                f"about 80% of real prices fall inside it (measured {m['interval_coverage_pct']:.0f}% on test data). "
                "The Model Evaluation page has the full breakdown.",
                slots,
            )
        if _HOW.search(message):
            return Reply(
                "I read the details from your message, fill in sensible defaults for anything optional, and run a "
                "LightGBM model trained on 6,000 listings across Bangalore, Mumbai and Pune. You get a most-likely "
                "price, a calibrated range, and the factors that moved the price most.",
                slots,
            )
        if looks_like_support_query(message):
            return Reply(
                "That sounds like a support question. The Help & Support page can answer questions on stamp duty, "
                "loans, RERA, accuracy and your account, or raise a ticket with our team.",
                slots,
            )
        return Reply(
            "I could not find property details in that. Try something like: 3 BHK flat in Powai, 1200 sq.ft, "
            "semi-furnished, with parking.",
            slots,
        )

    merged = {**slots, **fields}
    if city and "locality" not in fields:
        loc = merged.get("locality")
        if loc and LOCALITIES[loc].city != city:
            merged.pop("locality")
        merged["_city"] = city
    if "locality" in merged:
        merged["_city"] = LOCALITIES[merged["locality"]].city

    # Required: locality, and area or BHK
    if "locality" not in merged:
        known_city = merged.get("_city")
        if known_city:
            options = ", ".join(localities_for(known_city))
            return Reply(f"Which locality in {known_city}? I cover: {options}.", merged, used_llm=used_llm)
        return Reply(
            "Which area is the property in? I cover 30 localities across Bangalore, Mumbai and Pune, for example "
            "Whitefield, Koramangala, Powai, Andheri West, Baner or Hinjewadi.",
            merged,
            used_llm=used_llm,
        )
    if "area_sqft" not in merged and "bhk" not in merged:
        return Reply(
            f"Got it, {merged['locality']}. How big is it? Give the area in sq.ft or the number of bedrooms (BHK).",
            merged,
            used_llm=used_llm,
        )

    assumptions: list[str] = []
    values = {k: v for k, v in merged.items() if not k.startswith("_")}
    if "area_sqft" not in values:
        values["area_sqft"] = typical_area_for_bhk(values["bhk"])
        assumptions.append(f"area of {values['area_sqft']:,} sq.ft (typical for {values['bhk']} BHK)")
    if "bhk" not in values:
        values["bhk"] = typical_bhk_for_area(values["area_sqft"])
        assumptions.append(f"{values['bhk']} BHK (typical for {values['area_sqft']:,.0f} sq.ft)")
    if "bathrooms" not in values:
        values["bathrooms"] = values["bhk"]
        assumptions.append(f"{values['bathrooms']} bathrooms")
    defaults = PropertyInput.model_fields
    if "property_age" not in values:
        assumptions.append(f"{defaults['property_age'].default} years old")
    if "furnishing" not in values:
        assumptions.append(str(defaults["furnishing"].default).lower())
    if "parking" not in values:
        n = defaults["parking"].default
        assumptions.append(f"{n} parking space" + ("" if n == 1 else "s"))

    try:
        prop = PropertyInput(**values)
    except ValueError as exc:
        msg = str(exc).split("\n")[1] if "\n" in str(exc) else str(exc)
        return Reply(f"Some of those details are outside what I can price ({msg.strip()}). Could you check them?", merged)

    est = predictor.estimate(prop)
    top = est.top_drivers(3)
    drivers = "; ".join(f"{name} {'+' if pct >= 0 else ''}{pct:.0f}%" for name, pct in top)
    text = (
        f"Based on the trained model, a {prop.bhk} BHK {prop.property_type.lower()} of {prop.area_sqft:,.0f} sq.ft in "
        f"{prop.locality}, {prop.city} is most likely worth **{_fmt_lakhs(est.price_lakhs)}** "
        f"(likely range {_fmt_lakhs(est.low_lakhs)} to {_fmt_lakhs(est.high_lakhs)}), about "
        f"₹{est.price_per_sqft:,.0f} per sq.ft.\n\nBiggest factors versus an average listing: {drivers}."
    )
    if assumptions:
        text += "\n\nI assumed " + ", ".join(assumptions) + ". Tell me if any of that is different."
    return Reply(text, merged, est, prop, assumptions, used_llm)
