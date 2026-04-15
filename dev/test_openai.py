from llmai.openai import OpenAIClient
from llmai.shared.messages import TextContentPart, UserMessage


def test_generate():
    client = OpenAIClient()

    response = client.generate(
        model="gpt-5-mini",
        messages=[
            UserMessage(content=[TextContentPart(text="What is presentation?")]),
        ],
    )
    print(response)


test_generate()
