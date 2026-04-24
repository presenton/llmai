import unittest

from pydantic import ValidationError

from llmai.shared import (
    AssistantMessage,
    AssistantReasoningItem,
    ImageContentPart,
    ReasoningEffort,
    ReasoningEffortValue,
    ReasoningSummary,
    SystemMessage,
    TextContentPart,
    UserMessage,
    collapse_content_parts,
    flatten_thinking_content,
    normalize_content_parts,
)


class AssistantMessageTests(unittest.TestCase):
    def test_reasoning_effort_supports_effort_tokens_and_summary(self):
        reasoning = ReasoningEffort(
            effort=ReasoningEffortValue.HIGH,
            tokens=2048,
            summary=ReasoningSummary.DETAILED,
        )

        self.assertEqual(reasoning.effort, ReasoningEffortValue.HIGH)
        self.assertEqual(reasoning.tokens, 2048)
        self.assertEqual(reasoning.summary, ReasoningSummary.DETAILED)

    def test_reasoning_effort_rejects_negative_tokens(self):
        with self.assertRaises(ValidationError):
            ReasoningEffort(tokens=-1)

    def test_supports_optional_thinking_field(self):
        message = AssistantMessage(
            content=[TextContentPart(text="final answer")],
            thinking=[AssistantReasoningItem(summary=["hidden chain summary"])],
        )

        self.assertEqual(message.content[0].text, "final answer")
        self.assertEqual(flatten_thinking_content(message.thinking), ["hidden chain summary"])
        self.assertEqual(
            message.model_dump()["thinking"][0]["summary"],
            ["hidden chain summary"],
        )

    def test_supports_optional_assistant_message_id(self):
        message = AssistantMessage(
            id="msg_123",
            content=[TextContentPart(text="final answer")],
        )

        self.assertEqual(message.id, "msg_123")
        self.assertEqual(message.model_dump()["id"], "msg_123")

    def test_supports_optional_structured_thinking_items(self):
        message = AssistantMessage(
            content=[TextContentPart(text="final answer")],
            thinking=[
                AssistantReasoningItem(
                    id="rs_123",
                    summary=["hidden chain summary"],
                )
            ],
        )

        self.assertEqual(message.thinking[0].id, "rs_123")
        self.assertEqual(
            message.thinking[0].summary,
            ["hidden chain summary"],
        )

    def test_supports_multimodal_user_content(self):
        message = UserMessage(
            content=[
                TextContentPart(text="Describe this image."),
                ImageContentPart(url="https://example.com/cat.png"),
            ]
        )

        self.assertEqual(message.content[0].text, "Describe this image.")
        self.assertEqual(message.content[1].url, "https://example.com/cat.png")

    def test_supports_top_level_string_user_content(self):
        message = UserMessage(content="Describe this image.")

        content = normalize_content_parts(message.content)

        self.assertEqual(content[0].text, "Describe this image.")

    def test_supports_string_entries_in_user_content_lists(self):
        message = UserMessage(
            content=[
                "Describe this image.",
                ImageContentPart(url="https://example.com/cat.png"),
            ]
        )

        content = normalize_content_parts(message.content)

        self.assertEqual(content[0].text, "Describe this image.")
        self.assertEqual(content[1].url, "https://example.com/cat.png")

    def test_normalize_content_parts_converts_strings_to_text_parts(self):
        content = normalize_content_parts(
            [
                "Describe this image.",
                ImageContentPart(url="https://example.com/cat.png"),
            ]
        )

        self.assertEqual(content[0].text, "Describe this image.")
        self.assertEqual(content[1].url, "https://example.com/cat.png")

    def test_normalize_content_parts_converts_top_level_string_to_text_part(self):
        content = normalize_content_parts("Describe this image.")

        self.assertEqual(content[0].text, "Describe this image.")

    def test_supports_top_level_string_system_content(self):
        message = SystemMessage(content="Be concise.")

        self.assertEqual(message.content, "Be concise.")

    def test_rejects_non_string_system_content(self):
        with self.assertRaises(ValidationError):
            SystemMessage(content=["Be concise."])

    def test_collapse_content_parts_keeps_text_only_parts_as_a_list(self):
        content = collapse_content_parts(
            [
                TextContentPart(text="Hello"),
                TextContentPart(text=" world"),
            ]
        )

        self.assertEqual([part.text for part in content], ["Hello", " world"])

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
