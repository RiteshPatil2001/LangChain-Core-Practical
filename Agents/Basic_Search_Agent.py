from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq

load_dotenv()

search_tool = DuckDuckGoSearchRun()

model = ChatGroq(model="openai/gpt-oss-120b")

agent = create_agent(model=model, tools=[search_tool])

result = agent.invoke(
    {"messages": [("user", "What is today's top news in india?")]}
)

print(result["messages"][-1].content)