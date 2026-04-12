# To run this code you need to install the following dependencies:
# pip install google-genai

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()  # Load environment variables from .env file
client = genai.Client(
    api_key=os.getenv("GOOGLE_AI_API_KEY")
)

def classify(image_bytes, mime, model = "gemini-3.1-flash-lite-preview"):

    contents = [
        types.Content(
            role="system",
            parts=[
                types.Part.from_text(text="""You are an automated waste-sorting assistant. Look at the item in the image and classify it into exactly one of these three categories: 'garbage', 'recycling', or 'compost'"""),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime
                ),
            ],
        )
    ]
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_level="MINIMAL",
        ),
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type = genai.types.Type.OBJECT,
            required = ["Object Description", "Category"],
            properties = {
                "Object Description": genai.types.Schema(
                    description="A brief description of the object in the image.",
                    type = genai.types.Type.STRING,
                ),
                "Explanation": genai.types.Schema(
                    description="An explanation of why the object belongs to the identified category, in simple terms understanable by elementary school students.",
                    type = genai.types.Type.STRING,
                ),
                "Category": genai.types.Schema(
                    type = genai.types.Type.STRING,
                    enum = ["garbage", "recycling", "compost"],
                ),
            },
        ),
    )

    return client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    ).parsed
if __name__ == "__main__":
    url = "https://t4.ftcdn.net/jpg/04/98/36/27/360_F_498362712_7sJRmv7sOsfCtqieE0wtIjUpdUBvF4PY.jpg"
    import requests
    import tts
    response = requests.get(url)
    mime = response.headers['Content-Type']
    classification = classify(response.content, mime)

    tts.tts(classification["Explanation"])