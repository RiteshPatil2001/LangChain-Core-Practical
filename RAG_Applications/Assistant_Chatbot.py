from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import TextLoader

load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(
    model="Snowflake/snowflake-arctic-embed-l-v2.0"
)

parser = StrOutputParser()

text_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=800,
    chunk_overlap=100
)

llm = ChatGroq(model="openai/gpt-oss-120b")

prompt = PromptTemplate(
    template="""You are personal assistant chatbot\nYou only answer from provided context\nIf user asked question anything which is out of provided context then just simply say 'I don't know.'.\n\n context:- {context}\nQuestion:- {query}""",
    input_variables=["context", "query"]
)

try:
    loader = TextLoader("../files/myself.txt")
    docs = loader.load()
except FileNotFoundError:
    print('No Such PDF found on given location')
    exit()

chunks = text_splitter.split_documents(docs)

vector_store = FAISS.from_documents(chunks, embeddings)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 6}
)

query = input('\nEnter you query:- ')

retrieved_docs = retriever.invoke(query)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

final_prompt = prompt.invoke({"context":context_text, "query":query})

result = llm.invoke(final_prompt)

print(result.content)