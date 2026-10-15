from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b")

prompt_template = PromptTemplate(
    input_variables=["user_input", "word_count"],
    template="You are a medical expert. Answer about following disease: {user_input} in {word_count} words."
)

user_input = input("Enter disease name: ")
word_count = input("Enter word count: ")
prompt = prompt_template.invoke({ "user_input": user_input, "word_count": word_count })
response = model.invoke(prompt)
print("Response: ", response.content)
