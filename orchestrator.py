import streamlit as st
from intent_classifier import classify_intent
from document_qna import DocumentQnA
from web_qna import WebQnA
from database_qna import DatabaseQnA

class QnAOrchestrator:
    def __init__(self, groq_api_key: str, serp_api_key: str, pdf_path: str = None, db_path: str = "data.db", table_names: list = None):
        self.groq_api_key = groq_api_key
        self.serp_api_key = serp_api_key
        self.doc_qna = DocumentQnA(pdf_path=pdf_path) if pdf_path else DocumentQnA()
        self.web_qna = WebQnA(serp_api_key, groq_api_key)
        self.db_qna = DatabaseQnA(db_path=db_path, table_names=table_names)

    def answer(self, question: str) -> dict:
        intent = classify_intent(question, self.groq_api_key)
        if intent == "DOCUMENT":
            return self.doc_qna.answer(question, self.groq_api_key)
        elif intent == "DATABASE":
            return self.db_qna.answer(question, self.groq_api_key)
        else:
            return self.web_qna.answer(question)