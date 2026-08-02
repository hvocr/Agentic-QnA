from sentence_transformers import SentenceTransformer
import streamlit as st

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('all-mpnet-base-v2')

def generate_embeddings(chunks: list):
    model = get_embedding_model()
    return model.encode(chunks, convert_to_numpy=True)

# Alias for document_qna
def get_embeddings(chunks: list):
    return generate_embeddings(chunks)