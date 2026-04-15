import unittest

from llmai.shared import AssistantMessage


class AssistantMessageTests(unittest.TestCase):
    def test_supports_optional_thinking_field(self):
        message = AssistantMessage(
            content="final answer",
            thinking="hidden chain summary",
        )

        self.assertEqual(message.content, "final answer")
        self.assertEqual(message.thinking, "hidden chain summary")
        self.assertEqual(message.model_dump()["thinking"], "hidden chain summary")


if __name__ == "__main__":
    unittest.main()
