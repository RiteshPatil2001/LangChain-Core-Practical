from langchain_groq import ChatGroq
from langchain_core.tools import tool
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b")

messages = []

@tool
def multiply(a: int, b:int) -> int:
    """Gives product of given 2 numbers a, b"""
    return a*b

model_with_tools = model.bind_tools([multiply])

query = HumanMessage("can you multiply 3 with 27")

messages.append(query)

toolCall = model_with_tools.invoke(messages)

messages.append(toolCall)

toolmessage = multiply.invoke(toolCall.tool_calls[0])

messages.append(toolmessage)

result = model_with_tools.invoke(messages)

print(result.content)
