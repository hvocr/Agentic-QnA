import os
import pickle
import pandas as pd
from pdf_extract import extract_pdf_pagewise
from chunking import chunk_paragraphs
from generate_embeddings import get_embeddings
from search_embeddings import get_top_k_chunks
from groq import Groq
import streamlit as st

class DocumentQnA:
    def __init__(self, pdf_path: str = None, chunks_csv: str = "doc_chunks.csv", embeddings_pkl: str = "doc_embeddings.pkl"):
        self.pdf_path = pdf_path
        self.chunks_csv = chunks_csv
        self.embeddings_pkl = embeddings_pkl
        self.chunks = None
        self.embeddings = None
        self.load_or_process()

    def load_or_process(self):
        if os.path.exists(self.chunks_csv) and os.path.exists(self.embeddings_pkl):
            df = pd.read_csv(self.chunks_csv)
            self.chunks = df.to_dict("records")
            with open(self.embeddings_pkl, "rb") as f:
                self.embeddings = pickle.load(f)
        elif self.pdf_path and os.path.exists(self.pdf_path):
            structured = extract_pdf_pagewise(self.pdf_path)
            self.chunks = chunk_paragraphs(structured)
            contents = [ch["content"] for ch in self.chunks]
            self.embeddings = get_embeddings(contents)
            pd.DataFrame(self.chunks).to_csv(self.chunks_csv, index=False)
            with open(self.embeddings_pkl, "wb") as f:
                pickle.dump(self.embeddings, f)
        else:
            st.warning("No PDF loaded or processed. Please upload a PDF first.")

    def answer(self, question: str, groq_api_key: str, top_k: int = 3) -> dict:
        if not self.chunks or self.embeddings is None:
            return {"answer": "No document loaded. Please upload and process a PDF.", "source": "Document", "chunks": [], "confidence": 0.0}
        top_chunks = get_top_k_chunks(question, self.chunks, self.embeddings, k=top_k)
        if not top_chunks:
            return {"answer": "No relevant information found in the document.", "source": "Document", "chunks": [], "confidence": 0.0}
        context = "\n\n".join([ch["content"] for ch, _ in top_chunks])
        client = Groq(api_key=groq_api_key)
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer using ONLY the provided context. If the answer is not in the context, say 'I don't know'."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.2,
            max_tokens=300
        )
        answer = response.choices[0].message.content
        return {"answer": answer, "source": "Document", "chunks": [ch for ch, _ in top_chunks], "confidence": 0.95}
