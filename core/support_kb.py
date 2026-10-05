"""Support knowledge base: load Markdown sections and retrieve with TF-IDF."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core import llm
from core.config import KB_DIR
from core.parser import looks_like_price_query

CONFIDENCE_THRESHOLD = 0.13
_HUMAN = re.compile(r"\b(human|person|agent|someone|representative|raise (a )?ticket|complain|complaint|call me|speak to|talk to)\b", re.I)
_THANKS = re.compile(r"^\s*(thanks|thank you|ok|okay|great|cool|got it)\b", re.I)
_GREETING = re.compile(r"^\s*(hi|hello|hey|namaste)\b", re.I)


_STOP = TfidfVectorizer(stop_words="english").get_stop_words()


def _stem(word: str) -> str:
    """Light suffix stripping so 'supported' matches 'support' and 'loans' matches 'loan'."""
    for suffix in ("ations", "ation", "ings", "ing", "ies", "ed", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def _tokenize(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP]


@dataclass(frozen=True)
class Section:
    topic: str
    title: str
    body: str
    keywords: str = ""  # indexed for retrieval, never shown


@dataclass
class SupportReply:
    text: str
    sources: list[str]
    confidence: float
    offer_ticket: bool = False
    used_llm: bool = False


def load_sections(kb_dir: Path = KB_DIR) -> list[Section]:
    sections = []
    for path in sorted(kb_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        topic_match = re.search(r"^#\s+(.+)$", text, re.M)
        topic = topic_match.group(1).strip() if topic_match else path.stem
        for block in re.split(r"^##\s+", text, flags=re.M)[1:]:
            title, _, body = block.partition("\n")
            keywords = " ".join(re.findall(r"^Keywords:\s*(.+)$", body, re.M))
            body = re.sub(r"^Keywords:.*$\n?", "", body, flags=re.M)
            sections.append(Section(topic, title.strip(), body.strip(), keywords))
    return sections


class KnowledgeBase:
    def __init__(self, kb_dir: Path = KB_DIR) -> None:
        self.sections = load_sections(kb_dir)
        # Titles are phrased as questions, so weight them more than bodies
        docs = [f"{s.title} {s.title} {s.topic} {s.keywords} {s.body}" for s in self.sections]
        self.vectorizer = TfidfVectorizer(
            tokenizer=_tokenize, token_pattern=None, lowercase=False, ngram_range=(1, 2), sublinear_tf=True
        )
        self.matrix = self.vectorizer.fit_transform(docs)

    def search(self, query: str, k: int = 3) -> list[tuple[Section, float]]:
        sims = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        order = np.argsort(-sims)[:k]
        return [(self.sections[i], float(sims[i])) for i in order]

    def respond(self, question: str, history: list[dict] | None = None) -> SupportReply:
        if _HUMAN.search(question):
            return SupportReply(
                "I can connect you with our support team. Fill in the short form below and you will get a ticket "
                "ID straight away; the team replies by email within 2 business days.",
                [],
                1.0,
                offer_ticket=True,
            )
        if _GREETING.search(question) and len(question.split()) <= 3:
            return SupportReply(
                "Hello. I can help with how estimates work, accuracy, home loans and EMI, stamp duty, RERA, "
                "documents to check, and your data. What would you like to know?",
                [],
                1.0,
            )
        if _THANKS.search(question) and len(question.split()) <= 4:
            return SupportReply("Glad to help. Anything else?", [], 1.0)

        hits = self.search(question)
        best_section, best_score = hits[0]

        if best_score < CONFIDENCE_THRESHOLD and looks_like_price_query(question):
            return SupportReply(
                "To get a price, use the Price Assistant or the Predict page. Describe the property there, for "
                "example: 2 BHK in Kharadi, 1100 sq.ft.",
                [],
                best_score,
            )

        relevant = [(s, sc) for s, sc in hits if sc >= CONFIDENCE_THRESHOLD * 0.75]
        if relevant and llm.is_enabled():
            answer = llm.answer_support(question, [(s.title, s.body) for s, _ in relevant], history)
            if answer:
                return SupportReply(answer, [s.title for s, _ in relevant], best_score, used_llm=True)

        if best_score < CONFIDENCE_THRESHOLD:
            return SupportReply(
                "I could not find a confident answer to that in our help articles. Would you like to raise a "
                "ticket with our support team?",
                [],
                best_score,
                offer_ticket=True,
            )
        return SupportReply(best_section.body, [best_section.title], best_score)
