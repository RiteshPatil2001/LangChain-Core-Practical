from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

loader = TextLoader("../files/myself.txt")

docs = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

result = text_splitter.split_text(docs[0].page_content)

print(result)