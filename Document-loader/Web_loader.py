from langchain_community.document_loaders import WebBaseLoader

url = "https://www.flipkart.com/google-pixel-10-frost-256-gb/p/itm180af25bcc197?pid=MOBHEXHRXDGMF8XZ&marketplace=FLIPKART&lid=LSTMOBHEXHRXDGMF8XZ4R1W9O&pageUID=1788846789099"
loader = WebBaseLoader(
    web_path=url
)

docs = loader.load()

print(len(docs))