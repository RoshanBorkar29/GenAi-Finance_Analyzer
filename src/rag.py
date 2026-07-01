# src/rag.py
import os 
import google.genai as genai
from dotenv import load_dotenv

class FinancialRAG:
    # ✅ FIX 1: Corrected to double underscores __init__
    def __init__(self, model_name="gemini-2.5-flash"):
        load_dotenv()
        # ✅ FIX 2: Correctly instantiate the Client object (Capitalized C)
        self.client = genai.Client()
        self.model_name = model_name

    def build_context(self, retrieved_docs):
        context = ""
        for doc in retrieved_docs:
            source = doc.metadata.get("source", "Unknown")
            doc_type = doc.metadata.get("type", "Unknown")

            context += (
                f"Source: {source}\n"
                f"Type: {doc_type}\n\n"
                f"{doc.page_content}\n\n"
                f"---------------------\n\n"
            )
        return context
        
    def build_prompt(self, question, context):
        prompt = f"""
        You are an expert Financial Analyst.

        Answer the user's question ONLY using the provided context.

        If the answer is not present in the context,
        reply:

        "I could not find the answer in the provided annual report."

        Context
        ========

        {context}

        ========

        Question:
        {question}

        Instructions:

        1. Answer clearly.
        2. Use bullet points whenever suitable.
        3. Mention important financial numbers exactly.
        4. Never make up information.
        5. If multiple sources mention the answer, combine them into one response.
        """
        return prompt

    def generate_answer(self, question, retrieved_docs):
        context = self.build_context(retrieved_docs)
        prompt = self.build_prompt(question, context)
        
        # ✅ FIX 3: Properly structured SDK generation call using the client instance
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response.text