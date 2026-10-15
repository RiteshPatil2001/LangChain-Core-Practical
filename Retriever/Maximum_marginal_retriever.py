from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
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

retriever = vectore_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2, "lambda_mult":0.6}
)

result = retriever.invoke("What is LangChain")

print(result)