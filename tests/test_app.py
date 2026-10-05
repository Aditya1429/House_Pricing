"""Smoke tests: every page renders without exceptions, and no emojis anywhere."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
PAGES = ["home", "predict", "assistant", "compare", "explore", "evaluation", "support"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_file(str(ROOT / "views" / f"{page}.py"), default_timeout=60)
    at.run()
    assert not at.exception, at.exception


def test_assistant_chat_flow():
    at = AppTest.from_file(str(ROOT / "views" / "assistant.py"), default_timeout=60)
    at.run()
    at.chat_input[0].set_value("3 BHK in Baner, 1400 sqft").run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "most likely worth" in text


def test_support_chat_flow():
    at = AppTest.from_file(str(ROOT / "views" / "support.py"), default_timeout=60)
    at.run()
    at.chat_input[0].set_value("what is the stamp duty in Pune").run()
    assert not at.exception
    assert any("7%" in m.value for m in at.markdown)


_EMOJI = re.compile(
    r"[\U0001F300-\U0001FAFF\u2600-\u27BF\U0001F000-\U0001F2FF\u2B50\u2B55\uFE0F]"
)


def test_no_emojis_in_source():
    offenders = []
    for path in list(ROOT.glob("**/*.py")) + list(ROOT.glob("knowledge_base/*.md")):
        if ".venv" in path.parts:
            continue
        if _EMOJI.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
