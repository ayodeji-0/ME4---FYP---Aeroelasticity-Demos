import streamlit as st
from chatbot.retrieval import get_chunks_with_model
from chatbot.summarizer import summarize_text

st.title("🔍 Compare Embedding Models")

query = st.text_input("Ask a question to compare models:")
topic = st.selectbox("Select topic:", ["cbt_flutter", "blisks"])

col1, col2 = st.columns(2)

if query:
    with col1:
        st.markdown("### 🟦 MiniLM")
        minilm_chunks = get_chunks_with_model(query, "minilm", topic)
        minilm_summary = summarize_text(" ".join(minilm_chunks), question=query)
        st.subheader("Answer:")
        st.write(minilm_summary)
        st.subheader("Chunks:")
        for i, ch in enumerate(minilm_chunks):
            st.markdown(f"{i+1}. {ch}")

    with col2:
        st.markdown("### 🟥 MPNet")
        mpnet_chunks = get_chunks_with_model(query, "mpnet", topic)
        mpnet_summary = summarize_text(" ".join(mpnet_chunks), question=query)
        st.subheader("Answer:")
        st.write(mpnet_summary)
        st.subheader("Chunks:")
        for i, ch in enumerate(mpnet_chunks):
            st.markdown(f"{i+1}. {ch}")
