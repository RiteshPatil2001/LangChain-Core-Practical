from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal, Optional

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b")

class Review(TypedDict):
    key_theme: Annotated[list[str], "The main theme of the review, e.g., 'performance', 'design', 'battery life'"]
    summary: Annotated[str, "A brief summary of the review, highlighting the overall sentiment and key points"]
    sentiment: Annotated[Literal["positive", "negative", "neutral"], "The overall sentiment of the review, e.g., 'positive', 'negative', 'neutral'"]
    pros: Annotated[Optional[list[str]], "A list of positive aspects mentioned in the review, if any"]
    cons: Annotated[Optional[list[str]], "A list of negative aspects mentioned in the review, if any"]

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("Upgraded to the base Galaxy S24 after running my old device into the ground, and honestly, it’s a mixed bag of absolute brilliance and minor annoyances. First off, the size is a total relief. In a world where every phone feels like a mini tablet, actually being able to reach the top corner of the screen with one hand without doing finger gymnastics feels revolutionary. The flat matte edges and amber yellow finish look sleek, and that screen? Stunningly bright and crisp, easily visible even when you're standing directly under a glaring summer sun. Performance-wise, it’s an absolute rocket. Apps open instantly, scrolling is buttery smooth thanks to the 120Hz adaptive refresh rate, and switching between heavy tasks causes zero stutters. Battery life is surprisingly decent for a smaller form factor; it comfortably gets me through a full day of heavy scrolling, Spotify streaming, and messaging, though heavy 3D gaming definitely drains the juice quickly and makes the back noticeably warm.Now, for the gripes. Let's talk about the Galaxy AI hype. Honestly? Most of it feels like a marketing gimmick you’ll play with for two days and completely forget about. Live translate is neat on paper, but the generative photo-editing tools often leave weird artifacting if you're trying to clear out complex backgrounds. Shutter lag on moving objects—like pets or kids running around—is still a lingering issue, meaning too many quick shots turn out blurry. Also, Samsung's stubborn insistence on packing the software with duplicate apps and bloatware right out of the box is genuinely exhausting.Is it a good phone? Absolutely. The seven years of software support gives me massive peace of mind, and the main camera captures gorgeous, detailed shots with great dynamic range when objects stay still. But at this price point, you’re paying heavily for an AI suite that isn't quite ready for prime time. If you want a compact flagship that just handles daily grind tasks like a champ, it's a solid pick. Just don't buy it purely for the AI marketing.")

print(result["sentiment"])
