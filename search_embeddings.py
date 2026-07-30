from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import streamlit as st
import numpy as np

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('all-mpnet-base-v2')

def search(query: str, embeddings: np.ndarray, chunks: list, top_k: int = 3) -> list:
    if not chunks or embeddings is None or len(chunks) != len(embeddings):
        return []
    model = get_embedding_model()
    query_emb = model.encode([query], convert_to_numpy=True)
    similarities = cosine_similarity(query_emb, embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [(chunks[i], similarities[i]) for i in top_indices]

# Compatibility alias for document usage
def get_top_k_chunks(query, chunks, embeddings, k=3):
    return search(query, embeddings, chunks, k)
