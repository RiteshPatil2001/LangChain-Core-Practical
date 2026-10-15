from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from  langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from typing import Literal
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b")

parser1 = StrOutputParser()

class sentiment(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(..., description="Tells the sentiment of review either positive or negative")

parser2 = PydanticOutputParser(pydantic_object=sentiment)

prompt = PromptTemplate(
    template="From the following user review please draw a sentiment of review: {review} \n {format_instruction}",
    input_variables=['review'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

sentiment_chain = prompt | model | parser2

prompt2 = PromptTemplate(
    template='Write a thank you note for this positive user review {review}',
    input_variables=['review']
)

prompt3 = PromptTemplate(
    template='request a user to raise support ticket request for this negative user review {review}',
    input_variables=['review']
)

conditional_chain = RunnableBranch(
    (lambda x:x.sentiment=='positive', prompt2 | model | parser1),
    (lambda y:y.sentiment=='negative', prompt3 | model | parser1),
    RunnableLambda(lambda x:'Sentiment is neighter positive nor negative')
)

final_chain = sentiment_chain | conditional_chain

result = final_chain.invoke({'review':'this is very beautiful phone'})

print(result)
