from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b")

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Please state 5 one liner facts about the following place: {place}.",
    input_variables=["place"],
)

chain = prompt | model | parser

result = chain.invoke({"place": "Eiffel Tower"})

print(result)
