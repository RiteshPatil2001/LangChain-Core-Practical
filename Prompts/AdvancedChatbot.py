from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b")

prompt_template = ChatPromptTemplate([
    ('system', 'You are a chatbot assistant on e-commerce website. Questions outside the scope of e-commerce should be answered with "I am sorry, I am not able to answer that question. I am only able to answer questions related to e-commerce."'),
    MessagesPlaceholder(variable_name="chathistory"),
    ('human', '{user_input}')
])

chathistory = []

with open('Prompts/chat_history.txt', encoding='utf-8') as f:
    chathistory.extend(f.readlines())

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print(chathistory)
        break

    prompt = prompt_template.invoke({"chathistory": chathistory, "user_input":user_input})
    model_response = model.invoke(prompt)
    chathistory.append(HumanMessage(content=user_input))
    chathistory.append(AIMessage(content=model_response.content))
    print("AI: ", model_response.content)
