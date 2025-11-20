import streamlit as st
from src.query_rag import query_rag

st.set_page_config(page_title="Local PDF RAG", layout="wide", page_icon="📄")

st.sidebar.empty()  # Hide sidebar by default
top_k = st.sidebar.selectbox("top_k:", [1, 2, 3, 4, 5, 6, 7, 10], index=3)
sim_thres = st.sidebar.slider("similarity_threshold:", min_value=0.0, max_value=1.0, value=0.3, step=0.05)

col1, col2 = st.columns(2)
results, sources = [], []

with col1:
    st.header("Ask a Question")
    query = st.text_input(
        label = "Enter your question:",
        label_visibility="hidden",
        placeholder="Enter your question here...",
    )

    if query:
        with st.spinner("Retrieving response..."):
            response_text, sources, results = query_rag(query, top_k=top_k, similarity_threshold=sim_thres)
            st.header("Response")
        st.markdown(response_text)

with col2:
    st.header("Retreived Sources")
    for i, (res, src) in enumerate(zip(results, sources)):
        st.markdown(f"{i+1}. **{src}**: <br>*{res.page_content[:500]}...*", unsafe_allow_html=True)
        st.markdown("---")

