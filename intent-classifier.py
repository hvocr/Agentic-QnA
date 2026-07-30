from groq import Groq
import streamlit as st

def classify_intent(question: str, groq_api_key: str) -> str:
    """
    Returns one of: 'DOCUMENT', 'WEB', 'DATABASE'
    """
    client = Groq(api_key=groq_api_key)
    prompt = f"""
You are an intent classifier. Given a user question, decide which system should answer it:

- DOCUMENT: questions about a specific uploaded document (e.g., "What does the paper say about X?").
- WEB: questions about current events, general knowledge, or external information (e.g., "Who won the World Cup?").
- DATABASE: questions that require structured data retrieval (e.g., "Show me all sales records above 1000").

Return only one of: DOCUMENT, WEB, DATABASE.

Question: {question}
Intent:
"""
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        intent = response.choices[0].message.content.strip().upper()
        if intent in ["DOCUMENT", "WEB", "DATABASE"]:
            return intent
        return "WEB"  # fallback
    except Exception as e:
        st.error(f"Intent classification failed: {e}")
        return "WEB"  # safe fallback
