import streamlit as st
from web_search import fetch_web_content
from chunk_plain_text import chunk_plain_text
from generate_embeddings import generate_embeddings
from search_embeddings import search
from groq import Groq

class WebQnA:
    def __init__(self, serp_api_key: str, groq_api_key: str):
        self.serp_api_key = serp_api_key
        self.groq_api_key = groq_api_key
        self.chunks = []
        self.embeddings = None
        self.last_query = ""

    def answer(self, question: str, top_k: int = 3) -> dict:
        # If we don't have content for this query, fetch it
        if question != self.last_query or not self.chunks:
            self.last_query = question
            combined_text = fetch_web_content(question, self.serp_api_key, max_pages=3)
            if not combined_text:
                return {"answer": "No web content could be retrieved.", "source": "Web", "chunks": [], "confidence": 0.0}
            self.chunks = chunk_plain_text(combined_text, chunk_size=500, overlap=50)
            self.embeddings = generate_embeddings(self.chunks)

        top_results = search(question, self.embeddings, self.chunks, top_k=top_k)
        if not top_results:
            return {"answer": "No relevant information found on the web.", "source": "Web", "chunks": [], "confidence": 0.0}
        context = "\n\n---\n\n".join([chunk for chunk, _ in top_results])
        client = Groq(api_key=self.groq_api_key)
        messages = [
            {"role": "system", "content": (
                "You are a concise assistant. Answer the question using the provided context. "
                "If the context contains multiple rankings, lists, or sources that disagree, "
                "summarize the most relevant one(s) and briefly note that the answer can vary "
                "by source or metric (e.g. downloads vs. active players). "
                "Only say the context does not provide enough information if it is genuinely "
                "unrelated to the question."
            )},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.2,
            max_tokens=300
        )
        answer = response.choices[0].message.content
        return {"answer": answer, "source": "Web", "chunks": [chunk for chunk, _ in top_results], "confidence": 0.9}