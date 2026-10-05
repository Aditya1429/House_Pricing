import streamlit as st

from core import llm, tickets
from ui.components import footer, get_kb, page_header

kb = get_kb()

SUGGESTED = [
    "How accurate are the estimates?",
    "What is the stamp duty in Mumbai?",
    "How much home loan can I get?",
    "What documents should I check before buying?",
    "I want to talk to a person",
]

st.session_state.setdefault("sp_messages", [])
st.session_state.setdefault("sp_show_ticket", False)

page_header(
    "How can we help?",
    "Ask about estimates, home loans, stamp duty, RERA or your data. If the assistant cannot help, it will pass you "
    "to our support team.",
    "Help & Support",
    "support_agent",
)

chat_tab, ticket_tab, status_tab = st.tabs(["Ask a question", "Talk to a person", "Check ticket status"])


def _handle(question: str) -> None:
    msgs = st.session_state["sp_messages"]
    history = [{"role": m["role"], "content": m["content"]} for m in msgs]
    msgs.append({"role": "user", "content": question})
    reply = kb.respond(question, history)
    msgs.append(
        {
            "role": "assistant",
            "content": reply.text,
            "sources": reply.sources,
            "question": question,
            "offer_ticket": reply.offer_ticket,
        }
    )
    if reply.offer_ticket:
        st.session_state["sp_show_ticket"] = True


def _ticket_form(form_key: str) -> None:
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in st.session_state["sp_messages"])
    with st.form(form_key, clear_on_submit=True):
        c1, c2 = st.columns(2)
        name = c1.text_input("Name")
        email = c2.text_input("Email")
        category = st.selectbox("Category", tickets.CATEGORIES)
        last_q = next((m["content"] for m in reversed(st.session_state["sp_messages"]) if m["role"] == "user"), "")
        message = st.text_area("How can we help?", value=last_q, height=120)
        attach = st.checkbox("Include this chat conversation", value=bool(transcript))
        if st.form_submit_button("Submit ticket", type="primary"):
            errors = tickets.validate(name, email, message)
            if errors:
                for e in errors:
                    st.error(e)
            else:
                tid = tickets.create_ticket(name, email, category, message, transcript if attach else "")
                st.session_state["sp_show_ticket"] = False
                st.session_state["sp_last_ticket"] = tid
                st.rerun()


with chat_tab:
    if tid := st.session_state.pop("sp_last_ticket", None):
        st.success(
            f"Ticket {tid} created. Our team will reply by email within 2 business days. Keep the ID to check its status.",
            icon=":material/check_circle:",
        )

    if not st.session_state["sp_messages"]:
        st.markdown("**Common questions**")
        with st.container(key="chips_sp", horizontal=True):
            for q in SUGGESTED:
                if st.button(q, key=f"sg_{q}"):
                    _handle(q)
                    st.rerun()

    for i, msg in enumerate(st.session_state["sp_messages"]):
        avatar = ":material/support_agent:" if msg["role"] == "assistant" else ":material/person:"
        box_key = f"{'umsg' if msg['role'] == 'user' else 'amsg'}_s{i}"
        with st.container(key=box_key), st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if msg.get("sources"):
                    st.caption("Source: " + "; ".join(msg["sources"]))
                fb = st.feedback("thumbs", key=f"sp_fb_{i}")
                if fb is not None and not msg.get("fb_logged"):
                    tickets.record_feedback("support", msg.get("question", ""), msg["content"], bool(fb))
                    msg["fb_logged"] = True
                    if fb == 0:
                        st.session_state["sp_show_ticket"] = True
                        st.rerun()

    if st.session_state["sp_show_ticket"]:
        with st.container(key="card_inline_ticket"):
            st.markdown("**Raise a ticket with our support team**")
            _ticket_form("inline_ticket")

    if q := st.chat_input("Type your question"):
        _handle(q)
        st.rerun()
    st.caption("Answers: " + ("Claude, grounded in our help articles." if llm.is_enabled() else "matched from our help articles."))

with ticket_tab:
    st.markdown("Our team replies by email within 2 business days.")
    _ticket_form("ticket_tab_form")

with status_tab:
    with st.form("status_form"):
        tid_in = st.text_input("Ticket ID", placeholder="PL-000001")
        if st.form_submit_button("Check status"):
            t = tickets.get_ticket(tid_in)
            if t is None:
                st.warning("No ticket found with that ID. Check the format, for example PL-000012.")
            else:
                st.markdown(f"**{t['ticket_id']}**: {t['status']}")
                st.caption(f"{t['category']}, opened {t['created_at'][:10]}")
                st.text(t["message"])

footer()
