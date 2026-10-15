from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv

load_dotenv()

docs = [
    Document(
        page_content="Regular walking boosts heart health and can reduce symptoms of depression.",
        metadata={"source": "H1"},
    ),
    Document(
        page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.",
        metadata={"source": "H2"},
    ),
    Document(
        page_content="Deep sleep is crucial for cellular repair and emotional regulation.",
        metadata={"source": "H3"},
    ),
    Document(
        page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.",
        metadata={"source": "H4"},
    ),
    Document(
        page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.",
        metadata={"source": "H5"},
    ),
    Document(
        page_content="The solar energy system in modern homes helps balance electricity demand.",
        metadata={"source": "I1"},
    ),
    Document(
        page_content="Python balances readability with power, making it a popular system design language.",
        metadata={"source": "I2"},
    ),
    Document(
        page_content="Photosynthesis enables plants to produce energy by converting sunlight.",
        metadata={"source": "I3"},
    ),
    Document(
        page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.",
        metadata={"source": "I4"},
    ),
    Document(
        page_content="Black holes bend spacetime and store immense gravitational energy.",
        metadata={"source": "I5"},
    ),
]

embeddings = HuggingFaceEndpointEmbeddings(
    model="Snowflake/snowflake-arctic-embed-l-v2.0"
)

vectore_store = Chroma.from_documents(
    embedding=embeddings,
    documents=docs
)

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectore_store.as_retriever(search_type="mmr", search_kwargs={"k": 2, "lambda_mult":1}),
    llm=ChatGroq(model="openai/gpt-oss-120b")
)

result = multiquery_retriever.invoke("How to improve energy levels and maintain balance.")

for doc in result:
    print(doc.page_content,"\n")