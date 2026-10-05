import pytest

from core.parser import looks_like_price_query, parse

CASES = [
    ("I want to buy a 2 BHK house with 900 sq.ft.", {"bhk": 2, "area_sqft": 900}),
    ("Looking for 3 bedrooms, 2 bathrooms in Whitefield with parking", {"bhk": 3, "bathrooms": 2, "locality": "Whitefield", "parking": 1}),
    ("1500 sqft apartment 5 years old in Koramangala", {"area_sqft": 1500, "locality": "Koramangala", "property_type": "Apartment", "property_age": 5}),
    ("3BHK near Baner around 1.4k sqft", {"bhk": 3, "area_sqft": 1400, "locality": "Baner"}),
    ("brand new villa in whitefield with pool", {"property_type": "Villa", "property_age": 0, "has_pool": True}),
    ("2 bed flat in bandra with no parking, 12th floor of 30 storey tower", {"bhk": 2, "locality": "Bandra West", "parking": 0, "floor": 12, "total_floors": 30}),
    ("koramangla 1,250 sq ft semi furnished", {"locality": "Koramangala", "area_sqft": 1250, "furnishing": "Semi-Furnished"}),
    ("price of flat in hinjawadi 1100 sqft 10 yrs old", {"locality": "Hinjewadi", "area_sqft": 1100, "property_age": 10}),
    ("2bhk in E-City near metro", {"bhk": 2, "locality": "Electronic City", "metro_distance_km": 0.5}),
    ("fully furnished 4 BHK independent house in Jayanagar", {"bhk": 4, "furnishing": "Fully Furnished", "property_type": "Independent House", "locality": "Jayanagar"}),
    ("unfurnished studio in Andheri 450 sqft", {"bhk": 1, "furnishing": "Unfurnished", "locality": "Andheri West", "area_sqft": 450}),
    ("three bhk in Thane, 2 car parking, gym", {"bhk": 3, "locality": "Thane West", "parking": 2, "has_gym": True}),
    ("ready to move 2 BHK in Kharadi", {"property_age": 0, "bhk": 2, "locality": "Kharadi"}),
    ("1200 square feet in Wakad built in 2016", {"area_sqft": 1200, "locality": "Wakad", "property_age": 10}),
    ("ground floor flat in HSR layout without lift", {"floor": 0, "locality": "HSR Layout", "has_lift": False}),
    ("2 bhk koregaon park", {"bhk": 2, "locality": "Koregaon Park"}),
    ("kp 1800 sqft", {"locality": "Koregaon Park", "area_sqft": 1800}),
    ("Worli sea view 3 BHK 2000 sq ft on 25th floor", {"locality": "Worli", "bhk": 3, "area_sqft": 2000, "floor": 25}),
    ("navi mumbai 1 bhk", {"locality": "Kharghar", "bhk": 1}),
    ("bungalow in yelahanka 3000 sqft", {"property_type": "Villa", "locality": "Yelahanka", "area_sqft": 3000}),
    ("2 km from metro, 2 BHK in Hebbal", {"metro_distance_km": 2.0, "locality": "Hebbal"}),
]


@pytest.mark.parametrize("text,expected", CASES)
def test_extracts_expected_fields(text, expected):
    fields = parse(text).fields
    for key, value in expected.items():
        assert fields.get(key) == value, f"{key}: got {fields.get(key)!r} for {text!r}"


def test_plain_house_does_not_force_type():
    assert "property_type" not in parse("2 BHK house in Baner").fields


def test_city_detected_without_locality():
    r = parse("what is a 2 bhk worth in bengaluru")
    assert r.city == "Bangalore"
    assert "locality" not in r.fields


def test_unrelated_text_has_no_fields():
    assert parse("what is the weather like today").fields == {}


def test_price_intent():
    assert looks_like_price_query("price of 2bhk in baner")
    assert not looks_like_price_query("how do I delete my data")
