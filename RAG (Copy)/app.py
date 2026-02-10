import streamlit as st
from retrieval import ranked_docs
from llm import generate_answer


st.set_page_config(
    page_title="Indian Law Assistant",
    layout="centered"
)

st.title("Indian Law Assistant")
st.write("Ask questions based strictly on indexed legal documents.")

query = st.text_area(
    "Enter your legal question",
    placeholder="e.g. What is the punishment under IPC Section 420?",
    height=120
)


if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching legal documents..."):
            reranked_docs = ranked_docs(query)

        with st.spinner("Generating answer..."):
            answer = generate_answer(
                query=query,
                reranked_docs=reranked_docs
            )

        st.subheader("Answer")
        st.write(answer)
