from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b")

response = model.invoke("What is python?")

print(response.content)
