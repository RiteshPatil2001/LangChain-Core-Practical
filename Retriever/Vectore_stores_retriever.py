from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

docs = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM based search."),
    Document(page_content="Mumbai Indians (MI) are known for their strong squad and successful IPL history."),
    Document(page_content="The Indian subcontinent is a physiographic region of Asia south of the Himalayas which projects into the Indian Ocean between the Bay of Bengal to the east and the Arabian Sea to the west."),
]

embeddings = HuggingFaceEndpointEmbeddings(
    model="Snowflake/snowflake-arctic-embed-l-v2.0"
)

vectore_store = Chroma.from_documents(
    embedding=embeddings,
    documents=docs,
    persist_directory="chroma_db",
    collection_name="sample"
)

retriever = vectore_store.as_retriever(search_kwargs={"k": 2})

result = retriever.invoke("What is LLM.")

print(result)