from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b")

user_input = input("Enter your prompt: ")
response = model.invoke(user_input)
print("Response: ", response.content)
