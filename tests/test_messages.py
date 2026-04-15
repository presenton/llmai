import unittest

from pydantic import ValidationError

from llmai.shared import (
    AssistantMessage,
    ImageContentPart,
    TextContentPart,
    UserMessage,
    collapse_content_parts,
)


class AssistantMessageTests(unittest.TestCase):
    def test_supports_optional_thinking_field(self):
        message = AssistantMessage(
            content="final answer",
            thinking="hidden chain summary",
        )

        self.assertEqual(message.content, "final answer")
        self.assertEqual(message.thinking, "hidden chain summary")
        self.assertEqual(message.model_dump()["thinking"], "hidden chain summary")

    def test_supports_multimodal_user_content(self):
        message = UserMessage(
            content=[
                TextContentPart(text="Describe this image."),
                ImageContentPart(url="https://example.com/cat.png"),
            ]
        )

        self.assertEqual(message.content[0].text, "Describe this image.")
        self.assertEqual(message.content[1].url, "https://example.com/cat.png")

    def test_collapse_content_parts_returns_string_for_text_only_parts(self):
        content = collapse_content_parts(
            [
                TextContentPart(text="Hello"),
                TextContentPart(text=" world"),
            ]
        )

        self.assertEqual(content, "Hello world")

    def test_rejects_image_parts_without_exactly_one_source(self):
        with self.assertRaises(ValidationError):
            ImageContentPart()

        with self.assertRaises(ValidationError):
            ImageContentPart(
                url="https://example.com/cat.png",
                data=b"abc",
                mime_type="image/png",
            )

    def test_rejects_inline_image_without_mime_type(self):
        with self.assertRaises(ValidationError):
            ImageContentPart(data=b"abc")


if __name__ == "__main__":
    unittest.main()
