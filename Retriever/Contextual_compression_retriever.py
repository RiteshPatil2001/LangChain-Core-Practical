from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from dotenv import load_dotenv

load_dotenv()

docs = [
    Document(
        page_content=(
            """The Grand Canyon is one of the most visited natural wonders in the world. Photosynthesis is the process by which green plants convert sunlight into energy. Millions of tourists travel to see it every year. The rocks date back millions of years."""
        ),
        metadata={"source": "Doc1"},
    ),
    Document(
        page_content=(
            """In medieval Europe, castles were built primarily for defense. The chlorophyll in plant cells captures sunlight during photosynthesis. Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
        ),
        metadata={"source": "Doc2"},
    ),
    Document(
        page_content=(
            """Basketball was invented by Dr. James Naismith in the late 19th century. It was originally played with a soccer ball and peach baskets. NBA is now a global league."""
        ),
        metadata={"source": "Doc3"},
    ),
    Document(
        page_content=(
            """The history of cinema began in the late 1800s. Silent films were the earliest form. Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells. Modern filmmaking involves complex CGI and sound design."""
        ),
    ),
]

embeddings = HuggingFaceEndpointEmbeddings(
    model="Snowflake/snowflake-arctic-embed-l-v2.0"
)

vectore_store = Chroma.from_documents(
    embedding=embeddings,
    documents=docs,
)

bas_retriever = vectore_store.as_retriever(search_kwargs={"k": 3})

model = ChatGroq(model="openai/gpt-oss-120b")

compressor = LLMChainExtractor.from_llm(model)

compressor_retriever = ContextualCompressionRetriever(
    base_retriever=bas_retriever,
    base_compressor=compressor
)

result = compressor_retriever.invoke("What is photosynthesis")

for doc in result:
    print(doc.page_content,"\n")