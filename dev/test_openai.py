from dev.shared import SLIDE_SCHEMA
from llmai.openai import OpenAIClient
from llmai.shared.messages import UserMessage
from llmai.shared.response_formats import JSONSchemaResponse


def test_generate():
    client = OpenAIClient()

    response = client.generate(
        model="gpt-5-mini",
        messages=[
            UserMessage(content="What is presentation?"),
        ],
    )
    print("OpenAI plain generation")
    print(response)
    print("-" * 50)


def test_generate_structured():
    client = OpenAIClient()

    response = client.generate(
        model="gpt-5-mini",
        messages=[
            UserMessage(content="What is presentation?"),
        ],
        response_format=JSONSchemaResponse(
            name="ResponseSchema",
            strict=True,
            json_schema=SLIDE_SCHEMA,
        ),
    )
    print("OpenAI structured generation")
    print(response)
    print("-" * 50)


# test_generate()
test_generate_structured()
