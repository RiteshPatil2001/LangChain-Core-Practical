from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b")

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Write a funny joke about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="give explanation of the joke {text}",
    input_variables=['text']
)

# chain = prompt1 | model | parser | prompt2 | model | parser
chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

result = chain.invoke({'topic':'cricket'})

print(result)

