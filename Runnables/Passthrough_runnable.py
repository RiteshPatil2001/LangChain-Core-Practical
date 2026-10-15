from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

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

joke_gen_chain = prompt1 | model | parser

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': prompt2 | model | parser
})

chain = joke_gen_chain | parallel_chain

result = chain.invoke({'topic':'cricket'})

print(result)
