from __future__ import annotations

import unittest
from unittest.mock import patch

from tests.test_support import install_fake_torch_transformers

install_fake_torch_transformers()
from src.graph.nodes import critic as critic_mod


class FakeNode:
    def __init__(self, content: str) -> None:
        self._content = content

    def get_content(self) -> str:
        return self._content


class TestCriticShortAnswers(unittest.TestCase):
    def test_short_supported_answer_is_scored(self) -> None:
        docs = [FakeNode("The study reports that the answer is supported.")]

        with patch("src.graph.nodes.critic._batched_entailment_scores", return_value=[0.9]) as mocked:
            scores, feedback = critic_mod._sentence_support(["No"], docs)

        self.assertEqual(scores, [0.9])
        self.assertEqual(feedback, [])
        mocked.assert_called_once()


if __name__ == "__main__":
    unittest.main()
