
import os
import uuid
import streamlit as st

# Load API key from Streamlit Cloud Secrets
try:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    pass

from rag.hybrid_rag_pipeline import (
    get_rag_response,
    get_session_collection,
    process_uploaded_pdfs
)


# 1. Configure application
st.set_page_config(
    page_title="Hybrid RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Hybrid RAG Assistant")

st.write(
    "Upload your own PDF documents "
    "and ask questions about them."
)


# 2. Create a unique session ID
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex


# 3. Get this session's private collection
collection = get_session_collection(
    st.session_state.session_id
)


# 4. Upload PDFs
st.subheader("Upload your PDF documents")

uploaded_files = st.file_uploader(
    "Choose PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


# 5. Process uploaded PDFs
if st.button("Process PDFs"):

    if uploaded_files:

        try:
            with st.spinner("Processing your PDF documents..."):

                total_chunks = process_uploaded_pdfs(
                    uploaded_files,
                    collection
                )

            st.success(
                f"Successfully processed {len(uploaded_files)} PDF(s) "
                f"and stored {total_chunks} chunks!"
            )

        except Exception as error:
            st.error(f"PDF processing failed: {error}")

    else:
        st.warning("Please upload at least one PDF.")


# 6. Display document count
st.write("Stored document chunks:", collection.count())


# 7. Ask questions
st.subheader("Ask your documents")

question = st.text_input(
    "Enter your question:",
    placeholder="Ask something about your uploaded PDFs..."
)


# 8. Search
if st.button("Search"):

    if not question.strip():
        st.warning("Please enter a question.")

    elif collection.count() == 0:
        st.warning("Please upload and process PDFs first.")

    else:

        with st.spinner("Searching and generating answer..."):

            try:
                answer = get_rag_response(
                    question,
                    collection
                )

                st.subheader("AI Answer")
                st.write(answer)

            except Exception as error:
                st.error(f"Something went wrong: {error}")
