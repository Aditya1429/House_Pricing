"""Rule-based extraction of property attributes from natural language.

Handles phrasing such as "3BHK near Baner around 1.4k sqft, 5 yrs old",
"2 bed flat in bandra with no parking", or "brand new villa in Whitefield".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz, process

from core.config import CITY_ALIASES, LOCALITIES

_NUM = r"(\d+(?:[.,]\d+)*)\s*(k)?"
_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "single": 1, "double": 2, "a": 1, "an": 1, "no": 0, "zero": 0,
}

# alias (lowercase) -> canonical locality
_LOCALITY_ALIASES: dict[str, str] = {}
for _loc in LOCALITIES.values():
    _LOCALITY_ALIASES[_loc.name.lower()] = _loc.name
    for _a in _loc.aliases:
        _LOCALITY_ALIASES[_a] = _loc.name


def _to_number(raw: str, k: str | None = None) -> float:
    value = float(raw.replace(",", ""))
    return value * 1000 if k else value


def _word_or_digit(token: str) -> int | None:
    token = token.lower()
    if token.isdigit():
        return int(token)
    return _WORD_NUMBERS.get(token)


@dataclass
class ParseResult:
    fields: dict = field(default_factory=dict)
    city: str | None = None
    notes: list[str] = field(default_factory=list)

    def merge_into(self, slots: dict) -> dict:
        merged = {**slots, **self.fields}
        if self.city and "locality" not in self.fields:
            # A new city invalidates a locality from another city
            loc = merged.get("locality")
            if loc and LOCALITIES[loc].city != self.city:
                merged.pop("locality")
            merged["_city"] = self.city
        if "locality" in merged:
            merged["_city"] = LOCALITIES[merged["locality"]].city
        return merged


def find_locality(text: str) -> tuple[str | None, int]:
    t = text.lower()
    # Exact alias match, longest alias first so "bandra west" beats "bandra"
    for alias in sorted(_LOCALITY_ALIASES, key=len, reverse=True):
        if re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", t):
            return _LOCALITY_ALIASES[alias], 100
    # Fuzzy match on 1-3 word windows for typos like "koramangla" or "hinjawadi"
    words = re.findall(r"[a-z]+", t)
    windows = {" ".join(words[i : i + n]) for n in (1, 2, 3) for i in range(len(words) - n + 1)}
    windows = {w for w in windows if len(w) >= 4}
    best: tuple[str | None, int] = (None, 0)
    for w in windows:
        match = process.extractOne(w, list(_LOCALITY_ALIASES), scorer=fuzz.ratio)
        if match and match[1] > best[1]:
            best = (_LOCALITY_ALIASES[match[0]], int(match[1]))
    return best if best[1] >= 85 else (None, 0)


def parse(text: str) -> ParseResult:
    t = " " + text.lower().replace("sq. ft", "sqft").replace("sq.ft", "sqft").replace("sq ft", "sqft") + " "
    res = ParseResult()
    f = res.fields

    # Area
    m = re.search(_NUM + r"\s*(?:sqft|sq\s*\.?\s*feet|square\s*feet|sft|sq\b|square\s*foot|sqf)", t)
    if m:
        f["area_sqft"] = _to_number(m.group(1), m.group(2))
    else:
        m = re.search(r"(?:area|size|carpet|built[- ]?up)\D{0,12}" + _NUM, t)
        if m:
            f["area_sqft"] = _to_number(m.group(1), m.group(2))

    # Bedrooms
    m = re.search(r"(\d+(?:\.5)?|one|two|three|four|five|six|single|double)\s*-?\s*(?:bhk|bed\s*rooms?|bedrooms?|beds?\b|br\b|rk\b)", t)
    if m:
        token = m.group(1)
        f["bhk"] = int(float(token)) if token[0].isdigit() else _word_or_digit(token)
    elif re.search(r"\bstudio\b|\b1\s*rk\b", t):
        f["bhk"] = 1

    # Bathrooms
    m = re.search(r"(\d+|one|two|three|four|five|six)\s*(?:bath\s*rooms?|bathrooms?|baths?\b|toilets?|washrooms?)", t)
    if m:
        f["bathrooms"] = _word_or_digit(m.group(1))

    # Location
    loc, _score = find_locality(t)
    if loc:
        f["locality"] = loc
    for alias, city in CITY_ALIASES.items():
        if re.search(rf"\b{alias}\b", t):
            res.city = city
            break

    # Property type
    if re.search(r"\bvilla\b|\bbungalow\b|\browhouse\b|\brow house\b", t):
        f["property_type"] = "Villa"
    # Plain "house" is colloquial for any home in India, so it does not set a type
    elif re.search(r"independent\s*(?:house|home)|\bstandalone\b|\bbuilder floor\b|\bkothi\b", t):
        f["property_type"] = "Independent House"
    elif re.search(r"\bflat\b|\bapartment\b|\bcondo\b|\bstudio\b", t):
        f["property_type"] = "Apartment"

    # Furnishing
    if re.search(r"semi[- ]?furnish", t):
        f["furnishing"] = "Semi-Furnished"
    elif re.search(r"\bunfurnish|\bnot furnished|\bbare\s*shell", t):
        f["furnishing"] = "Unfurnished"
    elif re.search(r"fully[- ]?furnish|\bfurnished\b", t):
        f["furnishing"] = "Fully Furnished"

    # Parking
    if re.search(r"\bno\s*(?:car\s*)?parking\b|without\s*(?:a\s*)?parking", t):
        f["parking"] = 0
    else:
        m = re.search(r"(\d|one|two|three|single|double)\s*(?:car\s*)?(?:parking|parkings|car\s*spaces?|covered\s*parking)", t)
        if m:
            f["parking"] = min(3, _word_or_digit(m.group(1)) or 1)
        elif re.search(r"\bparking\b|\bcar\s*space\b|\bgarage\b", t):
            f["parking"] = 1

    # Age
    if re.search(r"brand\s*new|new\s*construction|under\s*construction|newly\s*built|\bnew\b\s*(?:flat|apartment|villa|house|project|property)|ready\s*to\s*move", t):
        f["property_age"] = 0
    else:
        m = re.search(r"(\d+)\s*(?:\+\s*)?(?:years?|yrs?)\s*(?:old)?", t)
        if m and re.search(r"old|age|built|construct", t):
            f["property_age"] = int(m.group(1))
        else:
            m = re.search(r"built\s*in\s*(19\d\d|20\d\d)", t)
            if m:
                f["property_age"] = max(0, 2026 - int(m.group(1)))

    # Floor
    m = re.search(r"(\d+)\s*(?:st|nd|rd|th)?\s*floor", t)
    if m:
        f["floor"] = int(m.group(1))
    elif re.search(r"ground\s*floor", t):
        f["floor"] = 0
    m = re.search(r"(?:of|out of|in a|in)\s*(?:a\s*)?(\d+)\s*(?:-?\s*)?(?:storey|story|floor)\s*(?:building|tower)?", t)
    if m and "floor" in f and int(m.group(1)) >= f["floor"]:
        f["total_floors"] = int(m.group(1))

    # Amenities
    if re.search(r"\bgym\b|\bfitness\b", t):
        f["has_gym"] = True
    if re.search(r"\bpool\b|swimming", t):
        f["has_pool"] = True
    if re.search(r"\bno\s*lift\b|\bwithout\s*(?:a\s*)?lift\b|\bno\s*elevator\b", t):
        f["has_lift"] = False
    elif re.search(r"\blift\b|\belevator\b", t):
        f["has_lift"] = True

    # Metro
    m = re.search(r"(\d+(?:\.\d+)?)\s*km\s*(?:from|to)?\s*(?:the\s*)?(?:metro|station)", t)
    if m:
        f["metro_distance_km"] = float(m.group(1))
    elif re.search(r"(?:near|next to|close to|walking distance (?:to|from))\s*(?:the\s*)?(?:metro|station)", t):
        f["metro_distance_km"] = 0.5

    return res


_PRICE_INTENT = re.compile(r"\b(price|cost|worth|value|estimate|valuation|how much|budget|buy|sell|bhk|sqft|flat|apartment|villa|house)\b")
_SUPPORT_INTENT = re.compile(r"\b(stamp duty|registration|rera|loan|emi|ticket|complaint|refund|account|password|privacy|contact|support|human|agent|accurate|accuracy|data source|how does|documents?)\b")


def looks_like_price_query(text: str) -> bool:
    t = text.lower()
    return bool(parse(t).fields) and bool(_PRICE_INTENT.search(t))


def looks_like_support_query(text: str) -> bool:
    return bool(_SUPPORT_INTENT.search(text.lower()))
