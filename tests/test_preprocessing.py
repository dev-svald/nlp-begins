"""
Unit tests for text preprocessing and cleaning.
"""

import unittest
from polarization.preprocessing import (
    calculate_text_length,
    clean_text,
    clean_text_strategy_1,
    clean_text_strategy_2,
)


class TestPreprocessing(unittest.TestCase):
    def test_url_removal(self):
        sample = "Check this breaking news https://t.co/xyz123 and http://example.com/test now!"
        cleaned1 = clean_text_strategy_1(sample)
        cleaned2 = clean_text_strategy_2(sample)

        self.assertNotIn("http", cleaned1)
        self.assertNotIn("https", cleaned1)
        self.assertNotIn("http", cleaned2)
        self.assertNotIn("https", cleaned2)
        self.assertIn("Check this breaking news", cleaned1)

    def test_user_mention_removal(self):
        sample = "Hey @johndoe and @sarah_99 look at this post"
        cleaned1 = clean_text_strategy_1(sample)
        cleaned2 = clean_text_strategy_2(sample)

        self.assertNotIn("@johndoe", cleaned1)
        self.assertNotIn("@sarah_99", cleaned1)
        self.assertNotIn("@johndoe", cleaned2)
        self.assertEqual(cleaned1, "Hey and look at this post")
        self.assertEqual(cleaned2, "Hey and look at this post")

    def test_whitespace_normalization(self):
        sample = "   Multiple    spaces   and \n newlines \t here   "
        cleaned = clean_text_strategy_1(sample)
        self.assertEqual(cleaned, "Multiple spaces and newlines here")

    def test_clean_text_dispatcher(self):
        sample = "Raw text @user https://test.org"
        self.assertEqual(clean_text(sample, strategy=0), sample)
        self.assertEqual(clean_text(sample, strategy=1), "Raw text")
        self.assertEqual(clean_text(sample, strategy=2), "Raw text")

    def test_calculate_text_length(self):
        sample = "One two three four five"
        self.assertEqual(calculate_text_length(sample), 5)


if __name__ == "__main__":
    unittest.main()
