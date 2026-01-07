import os
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from app.core.llm_loader import get_llm

class Contextualizer:
    def __init__(self):
        # On utilise Flash car ça doit être ultra-rapide
        self.llm = get_llm(temperature=0)
        
        template = """
        Your task is to rewrite the "Latest Question" into a standalone question that contains all necessary context.
        To do this, use the "Chat History" which contains the conversation between a Human and an AI Assistant.
        
        If the Latest Question refers to something the Assistant just said (e.g. "Is it expensive?", "Does it have warranty?"), 
        replace the pronoun with the specific object mentioned by the Assistant.

        Chat History:
        {history}

        Latest Question: {question}

        Standalone Question:"""
        
        prompt = PromptTemplate.from_template(template)
        self.chain = prompt | self.llm | StrOutputParser()

    def rewrite(self, history: str, question: str):
        if not history:
            return question
        try:
            return self.chain.invoke({"history": history, "question": question})
        except:
            return question

contextualizer = Contextualizer()

