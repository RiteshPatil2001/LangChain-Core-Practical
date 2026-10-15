from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableBranch, RunnableLambda

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b")

parser = StrOutputParser()

class Review(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(..., description="The overall sentiment of the review, e.g., 'positive', 'negative'")

parser2 = PydanticOutputParser(pydantic_object=Review)

prompt = PromptTemplate(
    template = "please draw and sentiment for the following review: {review}. \n {format_instructions}",
    input_variables=["review"],
    partial_variables={"format_instructions": parser2.get_format_instructions()}
)

sentiment_chain = prompt | model | parser2

prompt2 = PromptTemplate(
    template = "Please generate a response to the following positive review: {review}.",
    input_variables=["review"]
)

prompt3 = PromptTemplate(
    template = "Please generate a response to the following negative review: {review}.",
    input_variables=["review"]
)

positive_chain = prompt2 | model | parser

negative_chain = prompt3 | model | parser

conditional_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", positive_chain),
    (lambda x: x.sentiment == "negative", negative_chain),
    RunnableLambda(lambda x: "The sentiment is neither positive nor negative.")
)

final_chain = sentiment_chain | conditional_chain

text = """Returning my Galaxy S24 after 2 weeks. Disappointed.Rating: 2/5I bought the standard Galaxy S24 (Snapdragon version, 128GB) hoping for a smooth experience, but I ended up sending it back to Samsung. For a phone that costs this much, there are way too many compromises that ruin the daily experience.Here are the main issues I ran into:Shutter Lag and Motion Blur: The camera is a complete mess for anything moving. Trying to take a picture of my dog indoors turned into a fuzzy, blurry nightmare 8 times out of 10. Unless the lighting is 100% perfect and your subject stays completely still, the camera falls apart.  Display Grain at Low Brightness: I like reading in bed with the brightness turned down, but my panel had a super noticeable "grainy" texture on dark grey backgrounds. It makes the screen look dirty. Once you see it, you cant unsee it.  Aggressive Thermal Throttling: The phone gets uncomfortably warm fast. I took a few 4K videos at an outdoor barbecue, and within 10 minutes, the camera app gave me a temperature warning and capped my screen brightness so low I could barely see what I was recording.  RAM Management & App Rerenders: 8GB of RAM in a flagship phone is ridiculous. Browsing Reddit, switching to Maps for a route, and jumping back to Chrome consistently caused apps to reload from scratch.Verdict:It looks nice on paper, but the actual real-world optimization feels rushed. If you just browse social media, you might be fine, but for basic camera reliability and display quality at night, this was a hard skip for me."""

final_chain.get_graph().print_ascii()
