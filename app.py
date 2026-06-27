"""
app.py — Streamlit UI for the Personal Study Assistant.

Run with: streamlit run src/app.py

UI Layout:
  Sidebar:  Upload document, see status, clear cache
  Main:     Three tabs — Ask, Flashcards, Quiz

Design choices:
  - Session state holds the active file_hash (one doc at a time)
  - Flashcards render with a reveal toggle (front → click → back)
  - Quiz tracks score and shows explanations after submission
  - All heavy work (embedding) only happens once per document
"""

import sys
import os

# Allow imports from src/ whether run from project root or src/
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from loader import load_and_chunk
from vectorstore import embed_and_store, collection_exists
import rag


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Study Assistant",
    page_icon="📚",
    layout="wide",
)

print("hi")

# ── Session state init ─────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "file_hash": None,
        "doc_name": None,
        "chunks_count": 0,
        "revealed_cards": set(),
        "quiz_submitted": False,
        "quiz_answers": {},
        "quiz_data": None,
        "flashcard_data": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_state()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 Study Assistant")
    st.caption("RAG-powered. Grounded in your notes.")
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload your notes",
        type=["pdf", "md", "txt"],
        help="Supports PDF, Markdown, and plain text files.",
    )

    if uploaded_file:
        # Save uploaded file to data/ directory temporarily
        os.makedirs("data", exist_ok=True)
        save_path = os.path.join("data", uploaded_file.name)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Only re-process if it's a new file
        if st.session_state.doc_name != uploaded_file.name:
            with st.spinner("Loading and embedding document..."):
                chunks, file_hash = load_and_chunk(save_path)
                embed_and_store(chunks, file_hash)

                st.session_state.file_hash = file_hash
                st.session_state.doc_name = uploaded_file.name
                st.session_state.chunks_count = len(chunks)
                # Reset generated content when new doc loaded
                st.session_state.flashcard_data = None
                st.session_state.quiz_data = None
                st.session_state.quiz_submitted = False
                st.session_state.quiz_answers = {}
                st.session_state.revealed_cards = set()

    if st.session_state.file_hash:
        st.success(f"✓ {st.session_state.doc_name}")
        st.caption(f"{st.session_state.chunks_count} chunks indexed")

        if st.button("Clear document", use_container_width=True):
            for key in ["file_hash", "doc_name", "chunks_count",
                        "flashcard_data", "quiz_data"]:
                st.session_state[key] = None
            st.session_state.quiz_submitted = False
            st.session_state.quiz_answers = {}
            st.session_state.revealed_cards = set()
            st.rerun()
    else:
        st.info("Upload a document to get started.")

    st.divider()
    st.caption("**Stack:** Gemini 2.0 Flash · ChromaDB · all-MiniLM-L6-v2")


# ── Main area ──────────────────────────────────────────────────────────────────
if not st.session_state.file_hash:
    st.title("Personal Study Assistant")
    st.markdown("""
    **Upload your notes or a PDF in the sidebar to get started.**

    This assistant will:
    - Answer questions grounded in your documents
    - Generate flashcards from your notes
    - Quiz you with multiple-choice questions

    *No hallucinations — answers are pulled from your own content.*
    """)
    st.stop()

tab_qa, tab_flash, tab_quiz = st.tabs(["💬 Ask a Question", "🃏 Flashcards", "📝 Quiz"])


# ── Tab 1: Q&A ─────────────────────────────────────────────────────────────────
with tab_qa:
    st.header("Ask a Question")
    st.caption("Questions are answered using only your uploaded notes.")

    question = st.text_input(
        "Your question",
        placeholder="e.g. What is backpropagation and why does it work?",
        key="qa_input",
    )

    if st.button("Get Answer", type="primary", key="qa_button"):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Searching your notes..."):
                result = rag.ask_question(question, st.session_state.file_hash)

            st.markdown("### Answer")
            st.markdown(result["answer"][0]['text'])

            with st.expander("View retrieved chunks"):
                for i, chunk in enumerate(result["chunks"], 1):
                    section = chunk["metadata"].get("section", "(no section)")
                    source = chunk["metadata"].get("source", "")
                    sim = chunk["similarity"]
                    st.markdown(f"**Chunk {i}** | Section: `{section}` | Similarity: `{sim}`")
                    st.text(chunk["content"][:400] + ("..." if len(chunk["content"]) > 400 else ""))
                    if i < len(result["chunks"]):
                        st.divider()


# ── Tab 2: Flashcards ──────────────────────────────────────────────────────────
with tab_flash:
    st.header("Flashcards")
    st.caption("Click a card to reveal the answer.")

    col1, col2 = st.columns([2, 1])
    with col1:
        num_cards = st.slider("Number of cards", 3, 10, 5, key="num_cards")
    with col2:
        generate_btn = st.button("Generate Flashcards", type="primary", key="gen_flash")

    if generate_btn:
        with st.spinner("Generating flashcards from your notes..."):
            st.session_state.flashcard_data = rag.generate_flashcards(
                st.session_state.file_hash, num_cards=num_cards
            )
            st.session_state.revealed_cards = set()

    if st.session_state.flashcard_data:
        cards = st.session_state.flashcard_data.cards

        # Reset all / Reveal all controls
        ctrl_col1, ctrl_col2 = st.columns(2)
        with ctrl_col1:
            if st.button("Reveal All", key="reveal_all"):
                st.session_state.revealed_cards = set(range(len(cards)))
                st.rerun()
        with ctrl_col2:
            if st.button("Hide All", key="hide_all"):
                st.session_state.revealed_cards = set()
                st.rerun()

        st.divider()

        for i, card in enumerate(cards):
            is_revealed = i in st.session_state.revealed_cards
            with st.container():
                st.markdown(f"**Card {i + 1} of {len(cards)}**")
                st.info(f"**Q:** {card.front}")
                if is_revealed:
                    st.success(f"**A:** {card.back}")
                    if st.button(f"Hide answer", key=f"hide_{i}"):
                        st.session_state.revealed_cards.discard(i)
                        st.rerun()
                else:
                    if st.button(f"Reveal answer", key=f"reveal_{i}"):
                        st.session_state.revealed_cards.add(i)
                        st.rerun()
                st.divider()


# ── Tab 3: Quiz ────────────────────────────────────────────────────────────────
with tab_quiz:
    st.header("Quiz")
    st.caption("Multiple choice. Explanations shown after submission.")

    col1, col2 = st.columns([2, 1])
    with col1:
        num_questions = st.slider("Number of questions", 2, 8, 3, key="num_q")
    with col2:
        gen_quiz_btn = st.button("Generate Quiz", type="primary", key="gen_quiz")

    if gen_quiz_btn:
        with st.spinner("Generating quiz from your notes..."):
            st.session_state.quiz_data = rag.generate_quiz(
                st.session_state.file_hash, num_questions=num_questions
            )
            st.session_state.quiz_submitted = False
            st.session_state.quiz_answers = {}
        st.rerun()

    if st.session_state.quiz_data:
        quiz = st.session_state.quiz_data
        submitted = st.session_state.quiz_submitted

        for i, q in enumerate(quiz.questions):
            st.markdown(f"**Q{i + 1}: {q.question}**")
            options_map = {opt.label: opt.text for opt in q.options}
            option_labels = [f"{opt.label}. {opt.text}" for opt in q.options]

            if not submitted:
                selected = st.radio(
                    f"Options for Q{i + 1}",
                    option_labels,
                    key=f"q_{i}",
                    label_visibility="collapsed",
                )
                if selected:
                    st.session_state.quiz_answers[i] = selected[0]  # "A", "B", "C", or "D"
            else:
                # Show results
                user_ans = st.session_state.quiz_answers.get(i, "")
                is_correct = user_ans == q.correct

                for opt in q.options:
                    prefix = ""
                    if opt.label == q.correct:
                        prefix = "✅ "
                    elif opt.label == user_ans and not is_correct:
                        prefix = "❌ "
                    st.markdown(f"{prefix}**{opt.label}.** {opt.text}")

                if is_correct:
                    st.success("Correct!")
                else:
                    st.error(f"Incorrect. Correct answer: **{q.correct}**")
                st.caption(f"Explanation: {q.explanation}")

            st.divider()

        # Submit / Retry button
        if not submitted:
            if st.button("Submit Quiz", type="primary"):
                if len(st.session_state.quiz_answers) < len(quiz.questions):
                    st.warning("Answer all questions before submitting.")
                else:
                    st.session_state.quiz_submitted = True
                    st.rerun()
        else:
            # Score
            correct_count = sum(
                1 for i, q in enumerate(quiz.questions)
                if st.session_state.quiz_answers.get(i) == q.correct
            )
            total = len(quiz.questions)
            pct = int(correct_count / total * 100)
            st.metric("Your Score", f"{correct_count}/{total}", f"{pct}%")

            if st.button("Try Again (new quiz)", type="secondary"):
                st.session_state.quiz_submitted = False
                st.session_state.quiz_answers = {}
                st.session_state.quiz_data = None
                st.rerun()
