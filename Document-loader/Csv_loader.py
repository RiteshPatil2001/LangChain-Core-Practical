from langchain_community.document_loaders import CSVLoader

loader = CSVLoader("../files/IPL2025Batters.csv")

docs = loader.load()

print("\n\n--------------------------------------------------------------------\n\n")
print(docs)

# docs = loader.lazy_load()

# for document in docs:
#     print(document.page_content)