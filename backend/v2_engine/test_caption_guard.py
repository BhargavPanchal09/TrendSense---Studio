import unittest
from v2_engine.caption_guard import short_visual_caption


class CaptionGuardTests(unittest.TestCase):
    def test_returns_only_first_sentence(self):
        self.assertEqual(short_visual_caption("It is an IPO infographic. It has an imaginary factory and investor's house."),
                         "It is an IPO infographic.")

    def test_limits_caption_length(self):
        result = short_visual_caption(" ".join(["visible"] * 30) + ". Another sentence.")
        self.assertLessEqual(len(result.replace("...", "").split()), 24)
        self.assertNotIn("Another", result)

    def test_handles_empty_model_output(self):
        self.assertEqual(short_visual_caption("  "), "No image description was returned.")

    def test_placeholder_is_not_presented_as_visual_analysis(self):
        self.assertEqual(short_visual_caption('Image uploaded.'),
                         'The vision model did not provide a usable description.')


if __name__ == "__main__":
    unittest.main()
