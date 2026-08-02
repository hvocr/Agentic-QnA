from intent_classifier import classify_intent
from document_qna import DocumentQnA
from web_qna import WebQnA
from database_qna import DatabaseQnA
import os

class QnAOrchestrator:
    def __init__(self, groq_api_key: str, serp_api_key: str, pdf_path: str = None, db_path: str = "data.db", table_names: list = None):
        self.groq_api_key = groq_api_key
        self.serp_api_key = serp_api_key
        self.doc_qna = DocumentQnA(pdf_path=pdf_path) if pdf_path else DocumentQnA()
        self.web_qna = WebQnA(serp_api_key, groq_api_key)
        self.db_exists = os.path.exists(db_path)
        self.db_qna = DatabaseQnA(db_path=db_path, table_names=table_names) if self.db_exists else None

    def answer(self, question: str) -> dict:
        doc_loaded = bool(self.doc_qna.chunks)

        # Pass db_exists flag to classifier
        intent = classify_intent(question, self.groq_api_key, self.db_exists, doc_loaded)

        if intent == "DATABASE":
            if self.db_qna is not None:
                return self.db_qna.answer(question, self.groq_api_key)
            else:
                return {"answer": "Database not available. Please set up data.db.", "source": "Database", "chunks": [], "confidence": 0.0}

        # A document's topic is unpredictable, so we can't bias the classifier
        # with fixed keywords the way we do for the database. Instead, if a
        # document is loaded, always try it first regardless of what the
        # classifier guessed (WEB or DOCUMENT) — most questions about an
        # uploaded PDF don't explicitly say "in the document". Only fall
        # back to the web if the document genuinely has no answer.
        if doc_loaded:
            doc_result = self.doc_qna.answer(question, self.groq_api_key)
            no_answer_phrases = ("i don't know", "no relevant information", "no document loaded")
            if not any(p in doc_result["answer"].strip().lower() for p in no_answer_phrases):
                return doc_result

        if intent == "DOCUMENT" and not doc_loaded:
            return {"answer": "No document loaded. Please upload a PDF first.", "source": "Document", "chunks": [], "confidence": 0.0}

        return self.web_qna.answer(question)