from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(
    model="Snowflake/snowflake-arctic-embed-l-v2.0"
)

docs = [
    Document(
        page_content=(
            "Chennai Super Kings (CSK) are one of the most recognized teams in the IPL. "
            "MS Dhoni is important to CSK because of his leadership experience and long association with the franchise, "
            "while Ravindra Jadeja provides valuable all-round ability with both bat and ball."
        ),
        metadata={
            "team": "Chennai Super Kings",
            "short_name": "CSK",
            "players": ["MS Dhoni", "Ravindra Jadeja"]
        }
    ),

    Document(
        page_content=(
            "Mumbai Indians (MI) are known for their strong squad and successful IPL history. "
            "Rohit Sharma remains an important figure because of his long association and leadership legacy at MI, "
            "while Jasprit Bumrah is crucial for the team's bowling attack, especially in difficult phases of matches."
        ),
        metadata={
            "team": "Mumbai Indians",
            "short_name": "MI",
            "players": ["Rohit Sharma", "Jasprit Bumrah"]
        }
    ),

    Document(
        page_content=(
            "Royal Challengers Bengaluru (RCB) have a strong connection with Virat Kohli, who is one of the franchise's "
            "most important and recognizable players. Rajat Patidar is also important to the current RCB setup as a middle-order "
            "batter who can contribute aggressively during an innings."
        ),
        metadata={
            "team": "Royal Challengers Bengaluru",
            "short_name": "RCB",
            "players": ["Virat Kohli", "Rajat Patidar"]
        }
    ),

    Document(
        page_content=(
            "Rajasthan Royals (RR) have developed several young Indian players alongside experienced cricketers. "
            "Yashasvi Jaiswal is important because he provides explosive batting at the top of the order, "
            "while Riyan Parag has become an important all-round option for the team."
        ),
        metadata={
            "team": "Rajasthan Royals",
            "short_name": "RR",
            "players": ["Yashasvi Jaiswal", "Riyan Parag"]
        }
    ),

    Document(
        page_content=(
            "Kolkata Knight Riders (KKR) have relied on a combination of powerful batting and effective spin bowling. "
            "Sunil Narine is particularly important because he can contribute as both a spinner and a hard-hitting batter, "
            "while Rinku Singh is known for his ability to finish innings under pressure."
        ),
        metadata={
            "team": "Kolkata Knight Riders",
            "short_name": "KKR",
            "players": ["Sunil Narine", "Rinku Singh"]
        }
    ),

    Document(
        page_content=(
            "Sunrisers Hyderabad (SRH) are known for an aggressive style of batting and a strong top order. "
            "Pat Cummins is important because of his leadership and fast-bowling contribution, "
            "while Travis Head provides destructive batting and can change the momentum of a match quickly."
        ),
        metadata={
            "team": "Sunrisers Hyderabad",
            "short_name": "SRH",
            "players": ["Pat Cummins", "Travis Head"]
        }
    )
]

vectore_store = Chroma(
    embedding_function=embeddings,
    persist_directory="chroma_db",
    collection_name="sample"
)

vectore_store.add_documents(docs)

result = vectore_store.similarity_search(
    query="Tell me about Mumbai Indians.",
    k=1
)

# result = vectore_store.similarity_search_with_score(
#     query="Tell me about Mumbai Indians.",
#     k=1
# )

print(result)