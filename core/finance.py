"""Loan and purchase-cost arithmetic."""

from __future__ import annotations

# Approximate stamp duty + registration as a share of property value (2024-25).
# Actual rates vary by buyer gender, municipal area and property value.
STAMP_DUTY = {
    "Bangalore": {"stamp_duty_pct": 5.0, "registration_pct": 1.0, "note": "Karnataka: 5% stamp duty (above ₹45 L) + 1% registration"},
    "Mumbai": {"stamp_duty_pct": 6.0, "registration_pct": 0.0, "note": "Maharashtra (Mumbai): 6% incl. metro cess; registration capped at ₹30,000"},
    "Pune": {"stamp_duty_pct": 7.0, "registration_pct": 0.0, "note": "Maharashtra (Pune): 7% incl. cesses; registration capped at ₹30,000"},
}
REGISTRATION_CAP_MAHARASHTRA = 30000


def emi(principal: float, annual_rate_pct: float, years: int) -> float:
    """Monthly instalment for a reducing-balance loan."""
    n = years * 12
    if principal <= 0 or n <= 0:
        return 0.0
    r = annual_rate_pct / 12 / 100
    if r == 0:
        return principal / n
    return principal * r * (1 + r) ** n / ((1 + r) ** n - 1)


def max_loan_for_emi(monthly_emi: float, annual_rate_pct: float, years: int) -> float:
    n = years * 12
    r = annual_rate_pct / 12 / 100
    if r == 0:
        return monthly_emi * n
    return monthly_emi * ((1 + r) ** n - 1) / (r * (1 + r) ** n)


def affordability(monthly_income: float, existing_emi: float, down_payment: float, annual_rate_pct: float, years: int, foir: float = 0.5) -> dict:
    """Max property price given a fixed-obligation-to-income ratio (banks typically use 40-55%)."""
    available = max(0.0, monthly_income * foir - existing_emi)
    loan = max_loan_for_emi(available, annual_rate_pct, years)
    return {"max_emi": available, "max_loan": loan, "max_price": loan + down_payment}


def purchase_costs(price_rupees: float, city: str) -> dict:
    cfg = STAMP_DUTY[city]
    stamp = price_rupees * cfg["stamp_duty_pct"] / 100
    if cfg["registration_pct"]:
        registration = price_rupees * cfg["registration_pct"] / 100
    else:
        registration = min(price_rupees * 0.01, REGISTRATION_CAP_MAHARASHTRA)
    return {"stamp_duty": stamp, "registration": registration, "total": stamp + registration, "note": cfg["note"]}
