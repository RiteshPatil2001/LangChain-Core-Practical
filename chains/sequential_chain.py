from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b")

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Please generate an detailed report on the topic: {topic}.",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Please generate 5 one liner key points for the following report: {report}.",
    input_variables=["report"]
)

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"topic": "The impact of climate change on global agriculture"})

print(result)
