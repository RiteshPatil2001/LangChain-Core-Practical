from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path="../files",
    glob=["**.pdf"],
    loader_cls=PyPDFLoader
)

docs = loader.load()

print("\n\n--------------------------------------------------------------------\n\n")
print(docs)