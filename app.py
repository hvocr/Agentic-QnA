from document_qna import DocumentQnA
import streamlit as st
import os
from orchestrator import QnAOrchestrator
import traceback

st.set_page_config(page_title="Unified QnA", layout="wide")
st.title("🤖 Unified QnA – Document & Web")

# --- Secrets ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    SERP_API_KEY = st.secrets["SERPAPI_API_KEY"]
except KeyError as e:
    st.error(f"Missing secret: {e}. Please add it in Streamlit Cloud > Settings > Secrets.")
    st.stop()

# --- Sidebar: Upload PDF ---
with st.sidebar:
    st.header("📄 Upload Document (optional)")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    pdf_path = None
    if uploaded_file is not None:
        temp_pdf = "temp.pdf"
        with open(temp_pdf, "wb") as f:
            f.write(uploaded_file.read())
        st.success(f"Uploaded {uploaded_file.name}")
        pdf_path = temp_pdf
        st.session_state.pdf_path = pdf_path

# --- Initialize orchestrator ---
if "orchestrator" not in st.session_state:
    pdf_path = st.session_state.get("pdf_path", None)
    st.session_state.orchestrator = QnAOrchestrator(GROQ_API_KEY, SERP_API_KEY, pdf_path)
else:
    # Update PDF path if changed
    pdf_path = st.session_state.get("pdf_path", None)
    if pdf_path and st.session_state.orchestrator.doc_qna.pdf_path != pdf_path:
        st.session_state.orchestrator.doc_qna = DocumentQnA(pdf_path=pdf_path)

# --- Main area: Ask question ---
st.subheader("Ask anything")
question = st.text_input("Your question:")
if st.button("Get Answer", type="primary"):
    if not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.orchestrator.answer(question)
                st.subheader(f"✅ Answer (from {result['source']})")
                st.write(result["answer"])
                if result.get("chunks"):
                    with st.expander("📚 Retrieved evidence"):
                        for i, chunk in enumerate(result["chunks"]):
                            st.markdown(f"**Chunk {i+1}**")
                            st.write(chunk)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.code(traceback.format_exc())

st.info("This app uses intent detection to decide whether to answer from your document or from the web.")
