import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(page_title="Laws & Penalties Assistant", layout="wide")
st.title("⚖️ Laws & Penalties Assistant")

query = st.text_input("Ask a question about laws and penalties:")

if query:
    with st.spinner("Thinking..."):
        res = requests.post(API_URL, json={"question": query})
        data = res.json()

    st.subheader("Answer")
    st.write(data["answer"])

    with st.expander("Retrieved Evidence"):
        for i, c in enumerate(data["contexts"], 1):
            st.markdown(f"**Chunk {i}:**\n{c}")
