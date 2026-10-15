from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(
    model="Snowflake/snowflake-arctic-embed-l-v2.0"
)

vector = embeddings.embed_query("What is python?")

print(vector)
print("Embedding dimensions:", len(vector))
