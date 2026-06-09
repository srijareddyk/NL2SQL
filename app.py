
import os

import pandas as pd
import streamlit as st

from inference import DEMO_CSV, NL2SQLPipeline, load_schema_from_upload

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PASTEL_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Nunito', sans-serif;
    }

    .stApp {
        background: linear-gradient(165deg, #FFF8FA 0%, #FFE8F0 45%, #FFF0F6 100%);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFE4EE 0%, #FFD6E8 100%);
        border-right: 1px solid #F5C6D6;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #7A4A62 !important;
    }

    .hero-card {
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid #F5C6D6;
        border-radius: 20px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(232, 145, 173, 0.15);
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #7A4A62;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        color: #9A6B7E;
        font-size: 1.05rem;
        margin: 0;
    }

    .pill {
        display: inline-block;
        background: #FFD6E8;
        color: #7A4A62;
        border-radius: 999px;
        padding: 0.25rem 0.75rem;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }

    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #F8BBD9 0%, #E891AD 100%);
        color: #5C3D52;
        border: 1px solid #F0A8C4;
        border-radius: 12px;
        font-weight: 600;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }

    div[data-testid="stButton"] > button:hover {
        background: linear-gradient(135deg, #FFC9E0 0%, #F09BB8 100%);
        border-color: #E891AD;
        color: #4A2F3F;
        box-shadow: 0 4px 14px rgba(232, 145, 173, 0.35);
        transform: translateY(-1px);
    }

    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.65);
        border: 1px solid #F5D0DE;
        border-radius: 16px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.75rem;
    }

    [data-testid="stChatMessage"][data-testid*="user"],
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: #FFE4F0;
        border-color: #F0B8D0;
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: #FFF5F9;
        border-color: #F5D0DE;
    }

    [data-testid="stChatInput"] textarea {
        background: #FFF5F9 !important;
        border: 1px solid #F0B8D0 !important;
        border-radius: 14px !important;
        color: #5C3D52 !important;
    }

    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.55);
        border: 2px dashed #F0B8D0;
        border-radius: 14px;
        padding: 0.5rem;
    }

    div[data-baseweb="notification"].stAlert {
        border-radius: 12px;
    }

    [data-testid="stCodeBlock"] {
        border: 1px solid #F5D0DE;
        border-radius: 12px;
        overflow: hidden;
    }

    .empty-state {
        text-align: center;
        padding: 3rem 2rem;
        background: rgba(255, 255, 255, 0.6);
        border: 1px dashed #F0B8D0;
        border-radius: 20px;
        color: #9A6B7E;
    }

    .empty-state strong {
        color: #7A4A62;
    }
</style>
"""


@st.cache_resource(show_spinner="Loading T5 model and LoRA checkpoint…")
def get_pipeline(use_rag: bool) -> NL2SQLPipeline:
    return NL2SQLPipeline(use_rag=use_rag)


def render_header():
    st.markdown(
        """
        <div class="hero-card">
            <p class="hero-title">🌸 NL2SQL Chatbot</p>
            <p class="hero-subtitle">
                Ask questions in plain English about your CSV or SQLite file.
                The model generates SQL and runs it on your data.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="NL2SQL Chatbot", page_icon="🌸", layout="wide")
    st.markdown(PASTEL_CSS, unsafe_allow_html=True)
    render_header()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "schema" not in st.session_state:
        st.session_state.schema = None
    if "schema_label" not in st.session_state:
        st.session_state.schema_label = None

    with st.sidebar:
        st.markdown("### 📁 Data")
        uploaded = st.file_uploader(
            "Upload CSV or SQLite",
            type=["csv", "db", "sqlite", "sqlite3"],
        )
        if uploaded is not None:
            st.session_state.schema = load_schema_from_upload(uploaded.name, uploaded.getvalue())
            st.session_state.schema_label = uploaded.name
            st.session_state.messages = []
            t = st.session_state.schema["tables"][0]
            st.success(f"Loaded **{uploaded.name}**")
            st.markdown(
                f'<span class="pill">{len(t["columns"])} columns</span>'
                f'<span class="pill">{t["row_count"]} rows</span>',
                unsafe_allow_html=True,
            )
            with st.expander("Schema"):
                for c in t["columns"]:
                    samples = ", ".join(c["samples"][:3]) if c["samples"] else "—"
                    st.text(f"{c['name']} ({c['type']})  e.g. {samples}")

        st.divider()
        st.markdown("### ⚙️ Model")
        st.caption("Checkpoint: `checkpoints/best_lora/`")
        use_rag = st.toggle(
            "Enable RAG retrieval",
            value=True,
            help="Retrieve similar WikiSQL question/SQL pairs to augment the prompt.",
        )
        if use_rag:
            st.caption("Index: `data/processed/wikisql_retrieval/`")

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []

    if st.session_state.schema is None:
        st.markdown(
            """
            <div class="empty-state">
                <p><strong>Upload a file to get started</strong></p>
                <p>Use the sidebar to upload a CSV or SQLite file.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("retrieved_examples"):
                with st.expander(f"Retrieved examples ({len(msg['retrieved_examples'])})"):
                    for i, ex in enumerate(msg["retrieved_examples"], start=1):
                        st.markdown(f"**{i}.** {ex.question}")
                        st.code(ex.sql, language="sql")
            if msg.get("sql"):
                st.code(msg["sql"], language="sql")
            if isinstance(msg.get("result"), pd.DataFrame):
                st.dataframe(msg["result"], use_container_width=True)
            elif msg.get("result") is not None:
                st.warning(str(msg["result"]))

    question = st.chat_input("Ask a question about your table…")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Generating SQL…"):
                try:
                    pipeline = get_pipeline(use_rag)
                    out = pipeline.query(question, st.session_state.schema)
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"Error: {e}"}
                    )
                    return

                st.markdown("Here is the generated query and result:")
                retrieved = out.get("retrieved_examples") or []
                if retrieved:
                    with st.expander(f"Retrieved examples ({len(retrieved)})"):
                        for i, ex in enumerate(retrieved, start=1):
                            st.markdown(f"**{i}.** {ex.question}")
                            st.code(ex.sql, language="sql")
                if out["raw"] != out["sql"]:
                    with st.expander("Raw model output (WikiSQL struct)"):
                        st.text(out["raw"])
                st.code(out["sql"], language="sql")

                result = out["result"]
                if isinstance(result, pd.DataFrame):
                    st.dataframe(result, use_container_width=True)
                    summary = f"{len(result)} row(s) returned."
                else:
                    st.warning(str(result))
                    summary = str(result)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": summary,
                    "sql": out["sql"],
                    "result": result,
                    "retrieved_examples": retrieved,
                })


if __name__ == "__main__":
    main()
