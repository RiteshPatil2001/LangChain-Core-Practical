from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b")

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Generate a summarized report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a mcq quiz on {topic}',
    input_variables=['topic']
)

chain = RunnableParallel({
    'report': prompt1 | model | parser,
    'quiz': prompt2 | model | parser
})

result = chain.invoke({'topic':'Machine Learning'})

print(result['report'])
print('\n-------------------------------------------------------------------------------------------------------\n')
print(result['quiz'])