from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b")

prompt = [
    SystemMessage(content="You are a professional sports expert."),
    HumanMessage(content="Answer about following sport: Basketball.")
]

response = model.invoke(prompt)
prompt.append(AIMessage(content=response.content))
print("Response: ", response.content)