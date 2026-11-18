from rag.hybrid_rag_pipeline import get_rag_response
import streamlit as st
#  1. Configure the web application
st.set_page_config(
    page_title="Hybrid RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# 2. Display the application title
st.title("🤖 Hybrid RAG Assistant")

# 3. Display a short description
st.write(
    "Ask questions about your documents "
    "and get AI-powered answers."
)


# 4. Create a question input box
question = st.text_input(
    "Enter your question:",
    placeholder="Example: Who published this book?"
)

# 5. Create a Search button
search_button = st.button("Search")

# 6. Run the RAG pipeline when Search is clicked
if search_button:

    if question:

        with st.spinner("Searching documents and generating answer..."):

            answer = get_rag_response(question)

        st.subheader("AI Answer")
        st.write(answer)

    else:
        st.warning("Please enter a question.")


