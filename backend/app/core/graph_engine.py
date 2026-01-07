import os
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from app.core.llm_loader import get_llm

class GraphRAGEngine:
    def __init__(self):
        # Connexion Neo4j
        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI", "bolt://neo4j:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password1234")
        )
        
        # Rafraîchir le schéma
        self.graph.refresh_schema()
        self.llm = get_llm(temperature=0)
        
        # PROMPT SPECIAL : Force la recherche insensible à la casse
        cypher_generation_template = """
        Task: Generate Cypher statement to query a graph database.
        Instructions:
        1. Use only the provided schema.
        2. Do not use exact matching for strings. ALWAYS use `WHERE toLower(n.id) CONTAINS toLower('value')`.
        3. Do not include any explanations or apologies in your responses.
        4. If you cannot generate a query, simply return empty string.
        
        Schema:
        {schema}
        
        Question: {question}
        
        Cypher Query:"""
        
        cypher_prompt = PromptTemplate(
            template=cypher_generation_template,
            input_variables=["schema", "question"]
        )

        self.chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            cypher_prompt=cypher_prompt,
            verbose=True, # Affiche la requête Cypher dans les logs
            allow_dangerous_requests=True
        )

    def query(self, user_question: str):
        try:
            print(f"🕵️ GraphRAG cherche : {user_question}")
            response = self.chain.invoke(user_question)
            return response['result']
        except Exception as e:
            print(f"⚠️ Erreur GraphRAG: {e}")
            return None

graph_engine = GraphRAGEngine()
