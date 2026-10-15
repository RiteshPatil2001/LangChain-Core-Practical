from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

loader = TextLoader("../files/myself.txt")

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0
)

result = text_splitter.split_text(docs[0].page_content)

print("\n\n", len(result),"\n\n")
for spilt in result:
    print(spilt)
    print("\n---------------------------------------------------------------------------------\n\n")