from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b")

class Review(BaseModel):
    key_theme: list[str] = Field(..., description="The main theme of the review, e.g., 'performance', 'design', 'battery life'")
    summary: str = Field(..., description="A brief summary of the review, highlighting the overall sentiment and key points")
    sentiment: Literal["positive", "negative", "neutral"] = Field(..., description="The overall sentiment of the review, e.g., 'positive', 'negative', 'neutral'")
    pros: list[str] = Field(None, description="A list of positive aspects mentioned in the review, if any")
    cons: list[str] = Field(None, description="A list of negative aspects mentioned in the review, if any")

structured_model = model.with_structured_output(Review)

result = structured_model.invoke('Returning my Galaxy S24 after 2 weeks. Disappointed.Rating: 2/5I bought the standard Galaxy S24 (Snapdragon version, 128GB) hoping for a smooth experience, but I ended up sending it back to Samsung. For a phone that costs this much, there are way too many compromises that ruin the daily experience.Here are the main issues I ran into:Shutter Lag and Motion Blur: The camera is a complete mess for anything moving. Trying to take a picture of my dog indoors turned into a fuzzy, blurry nightmare 8 times out of 10. Unless the lighting is 100% perfect and your subject stays completely still, the camera falls apart.  Display Grain at Low Brightness: I like reading in bed with the brightness turned down, but my panel had a super noticeable "grainy" texture on dark grey backgrounds. It makes the screen look dirty. Once you see it, you cant unsee it.  Aggressive Thermal Throttling: The phone gets uncomfortably warm fast. I took a few 4K videos at an outdoor barbecue, and within 10 minutes, the camera app gave me a temperature warning and capped my screen brightness so low I could barely see what I was recording.  RAM Management & App Rerenders: 8GB of RAM in a flagship phone is ridiculous. Browsing Reddit, switching to Maps for a route, and jumping back to Chrome consistently caused apps to reload from scratch.Verdict:It looks nice on paper, but the actual real-world optimization feels rushed. If you just browse social media, you might be fine, but for basic camera reliability and display quality at night, this was a hard skip for me.')

print(result)