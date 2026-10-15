from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq
from langchain_core.tools import tool
import requests

load_dotenv()

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
    """This function fetches current weather data for a given city"""
    url = f'https://api.weatherstack.com/current?access_key=93d6f66be7d5c7cda0a26caa50f3e741&query={city}'
    response = requests.get(url, timeout=5)
    return response.json()

model = ChatGroq(model="openai/gpt-oss-120b")

agent = create_agent(model=model, tools=[search_tool, get_weather_data])

result = agent.invoke(
    {"messages": [("user", "What is todays weather of Mumbai")]}
)

print(result["messages"][-1].content)