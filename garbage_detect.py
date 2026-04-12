# To run this code you need to install the following dependencies:
# pip install google-genai

import os
from google import genai
from google.genai import types


def generate():
    client = genai.Client(
        api_key="AIzaSyCvj9ra6L-YiWQ0hK6CzOiWA4pR1_jKA1U",
    )

    model = "gemini-3.1-flash-lite-preview"
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
                types.Part.from_uri(file_uri="https://t4.ftcdn.net/jpg/04/98/36/27/360_F_498362712_7sJRmv7sOsfCtqieE0wtIjUpdUBvF4PY.jpg"),
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

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if text := chunk.text:
            print(text, end="")

if __name__ == "__main__":
    generate()