import pytest

from core import finance, tickets
from core.support_kb import KnowledgeBase


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase()


@pytest.mark.parametrize(
    "question,expected_title",
    [
        ("what is stamp duty in pune", "What is the stamp duty in Pune?"),
        ("stamp duty mumbai", "What is the stamp duty in Mumbai?"),
        ("how accurate are your estimates", "How accurate are the estimates?"),
        ("how is emi calculated", "How is EMI calculated?"),
        ("what is RERA", "What is RERA?"),
        ("how much loan can I get", "How much home loan can I get?"),
        ("can I delete my data", "How do I delete my data?"),
        ("carpet area vs super built up", "What is the difference between carpet, built-up and super built-up area?"),
        ("do you support delhi", "Which cities and localities are supported?"),
        ("why is the price different from what the seller is asking", "Why is my estimate different from the asking price?"),
    ],
)
def test_retrieval(kb, question, expected_title):
    r = kb.respond(question)
    assert r.sources and r.sources[0] == expected_title


def test_human_request_offers_ticket(kb):
    assert kb.respond("I want to talk to a human").offer_ticket


def test_unknown_question_offers_ticket(kb):
    r = kb.respond("what is the weather today")
    assert r.offer_ticket and not r.sources


def test_keywords_hidden_from_answers(kb):
    assert all("Keywords:" not in s.body for s in kb.sections)


def test_ticket_round_trip(tmp_path):
    db = tmp_path / "support.db"
    tid = tickets.create_ticket("Asha", "asha@example.com", "Bug report", "The map does not load.", db_path=db)
    assert tid == "PL-000001"
    t = tickets.get_ticket("pl-1", db_path=db)
    assert t["status"] == "Open" and t["category"] == "Bug report"
    assert tickets.get_ticket("PL-999", db_path=db) is None


def test_ticket_validation():
    assert tickets.validate("", "bad", "short") and len(tickets.validate("", "bad", "short")) == 3
    assert tickets.validate("Asha", "asha@example.com", "Please help with my estimate") == []


def test_feedback_logging(tmp_path):
    db = tmp_path / "support.db"
    tickets.record_feedback("support", "q", "a", True, db_path=db)
    tickets.record_feedback("support", "q", "a", False, db_path=db)
    assert tickets.feedback_summary(db_path=db) == {"total": 2, "helpful": 1}


def test_emi_known_value():
    # ₹50 L at 8.5% for 20 years is about ₹43,391 per month
    assert finance.emi(5_000_000, 8.5, 20) == pytest.approx(43391, abs=5)


def test_affordability_inverts_emi():
    a = finance.affordability(100000, 0, 1_000_000, 9.0, 20)
    assert finance.emi(a["max_loan"], 9.0, 20) == pytest.approx(a["max_emi"], rel=1e-6)


def test_maharashtra_registration_cap():
    c = finance.purchase_costs(20_000_000, "Mumbai")
    assert c["registration"] == 30000
