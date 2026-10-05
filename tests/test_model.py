import pytest

from core.assistant import respond
from core.predict import Predictor
from core.schema import PropertyInput


@pytest.fixture(scope="module")
def predictor():
    return Predictor()


def _p(**kw):
    base = dict(locality="Baner", area_sqft=1200, bhk=2, bathrooms=2)
    return PropertyInput(**{**base, **kw})


def test_price_rises_with_area(predictor):
    small = predictor.estimate(_p(area_sqft=800)).price_lakhs
    large = predictor.estimate(_p(area_sqft=2000, bhk=3, bathrooms=3)).price_lakhs
    assert large > small * 1.6


def test_premium_locality_costs_more(predictor):
    worli = predictor.estimate(_p(locality="Worli")).price_lakhs
    wagholi = predictor.estimate(_p(locality="Wagholi")).price_lakhs
    assert worli > wagholi * 5


def test_new_property_costs_more_than_old(predictor):
    new = predictor.estimate(_p(property_age=0)).price_lakhs
    old = predictor.estimate(_p(property_age=25)).price_lakhs
    assert new > old


def test_range_brackets_estimate(predictor):
    e = predictor.estimate(_p())
    assert e.low_lakhs <= e.price_lakhs <= e.high_lakhs
    assert e.high_lakhs / e.low_lakhs < 2


def test_estimate_close_to_locality_rate(predictor):
    # A typical 2 BHK apartment should land near the locality's per-sq.ft rate
    e = predictor.estimate(_p(locality="Whitefield", property_age=5))
    assert 7000 < e.price_per_sqft < 11500


def test_drivers_reconstruct_estimate(predictor):
    e = predictor.estimate(_p(locality="Powai"))
    product = e.baseline_lakhs
    for pct in e.drivers.values():
        product *= 1 + pct / 100
    assert product == pytest.approx(e.price_lakhs, rel=1e-3)


def test_comparables_same_city(predictor):
    comps = predictor.comparables(_p(locality="Powai"))
    assert len(comps) == 5
    from core.config import LOCALITIES

    assert all(LOCALITIES[l].city == "Mumbai" for l in comps["Locality"])


def test_conversation_slot_filling(predictor):
    r1 = respond("2 BHK with 900 sq.ft", {}, predictor)
    assert r1.estimate is None and "area" in r1.text.lower()
    r2 = respond("in Baner", r1.slots, predictor)
    assert r2.estimate is not None
    assert r2.property.area_sqft == 900 and r2.property.locality == "Baner"
    r3 = respond("make it 1100 sqft", r2.slots, predictor)
    assert r3.property.area_sqft == 1100 and r3.property.locality == "Baner"


def test_conversation_city_switch_asks_locality(predictor):
    r1 = respond("2 BHK in Baner 1000 sqft", {}, predictor)
    r2 = respond("actually in mumbai", r1.slots, predictor)
    assert r2.estimate is None and "Mumbai" in r2.text


def test_conversation_reset(predictor):
    r = respond("start over", {"locality": "Baner"}, predictor)
    assert r.slots == {}
