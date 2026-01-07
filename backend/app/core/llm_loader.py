import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

def get_llm(temperature=0):
    """
    Renvoie le modèle configuré dans le .env (Google ou Groq)
    """
    provider = os.getenv("LLM_PROVIDER", "google").lower()
    
    if provider == "groq":
        print(f"⚡ Utilisation de GROQ ({os.getenv('GROQ_MODEL')})")
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=temperature,
            api_key=os.getenv("GROQ_API_KEY")
        )
    
    else:
        print(f"🤖 Utilisation de GOOGLE ({os.getenv('GOOGLE_MODEL')})")
        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-1.5-flash"),
            temperature=temperature
        )