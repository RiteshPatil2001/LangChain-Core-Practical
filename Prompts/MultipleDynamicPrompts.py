from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b")

template = ChatPromptTemplate([
    ('system', '{user_role}'),
    ('human', '{user_input}')
])

user_role = input("Enter your Assistant role: ")
user_input = input("Enter your question: ")

prompt = template.invoke({"user_role": user_role, "user_input": user_input})

model_response = model.invoke(prompt)
print("Response: ", model_response.content)
