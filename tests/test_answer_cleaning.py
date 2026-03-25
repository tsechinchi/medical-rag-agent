from __future__ import annotations

import unittest

from src.utils.answer_cleaning import (
    INSUFFICIENT_EVIDENCE,
    clean_answer_text,
    clean_for_scoring,
    is_abstention,
)


class TestAnswerCleaning(unittest.TestCase):
    def test_clean_for_scoring_strips_safety_prefixes_and_disclaimers(self) -> None:
        raw = (
            "[Partially Supported] The treatment improved survival.[1]\n"
            "Medical disclaimer: consult your clinician.\n"
            "Sources: [1]"
        )

        self.assertEqual(clean_answer_text(raw, max_sentences=None), "The treatment improved survival.[1]")
        self.assertEqual(clean_for_scoring(raw), "The treatment improved survival.")

    def test_abstention_detection(self) -> None:
        self.assertTrue(is_abstention(INSUFFICIENT_EVIDENCE))


if __name__ == "__main__":
    unittest.main()
