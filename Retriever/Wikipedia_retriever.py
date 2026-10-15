from langchain_community.retrievers import WikipediaRetriever
import wikipedia

wikipedia.set_user_agent(
    "RiteshRAGTest/1.0 (ritesh@example.com)"
)
retriever = WikipediaRetriever(top_k_results=2, lang="en")

query = "Indian geopolitics"

docs = retriever.invoke(query)

print(docs)
