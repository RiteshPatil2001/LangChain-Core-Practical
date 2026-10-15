from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b")

chat_history = []

while True:
    user_input = input("You: ")
    chat_history.append(user_input)
    if user_input.lower() == 'exit':
        print(chat_history)
        break

    model_response = model.invoke(chat_history)
    chat_history.append(model_response.content)
    print("AI: ", model_response.content)