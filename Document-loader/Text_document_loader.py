from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

loader = TextLoader('../files/myself.txt')

docs = loader.load()

model = ChatGroq(model="openai/gpt-oss-120b")

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Here is brief about me:\n{myself} \n\n from above introduction give answers of following questions \n\n {question}",
    input_variables=['myself', 'question']
)

chain = prompt | model | parser

user_question = input("Enter your Question: ")

print(chain.invoke({'myself':docs[0].page_content, 'question': user_question}))