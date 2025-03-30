
import streamlit as st
from chatbot.retrieval import get_chunks_with_model
from chatbot.summarizer import summarize_text

st.title("🔍 Compare Embedding Models")

query = st.text_input("Ask a question to compare models:")
topic = st.selectbox("Select topic:", ["cbt_flutter", "blisks"])

# Select models to compare
selected_models = st.multiselect(
    "Choose embedding models to compare:",
    ["minilm", "mpnet", "instructor_xl"],
    default=["minilm", "mpnet"]
)

cols = st.columns(len(selected_models))

for i, model_name in enumerate(selected_models):
    with cols[i]:
        st.markdown(f"### 🔹 {model_name.upper()}")
        try:
            chunks = get_chunks_with_model(query, model_name, topic)
            summary = summarize_text(" ".join(chunks), question=query)
            st.subheader("Answer:")
            st.write(summary)
            st.subheader("Chunks:")
            for idx, ch in enumerate(chunks):
                st.markdown(f"{idx+1}. {ch}")
        except Exception as e:
            st.error(f"Error with model {model_name}: {e}")
