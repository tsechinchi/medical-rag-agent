from __future__ import annotations

import unittest

from tests.test_support import FakeDataFrame, install_fake_pandas

install_fake_pandas()
from src.evaluation.result_utils import align_eval_frames, safe_mean
from src.evaluation.run_metadata import compute_run_id


class TestResultUtils(unittest.TestCase):
    def test_align_eval_frames_masks_mismatched_runs(self) -> None:
        questions = ["q1", "q2"]
        meta = {"profile": "fast", "budget_seconds": 10800}
        run_id = compute_run_id(questions, meta)

        model_free = FakeDataFrame(
            {
                "question": questions,
                "run_id": [run_id, run_id],
                "abstention_detected": [0, 1],
                "answer_relevancy": [0.8, 0.0],
            }
        )
        bert = FakeDataFrame(
            {
                "question": questions,
                "run_id": [run_id, run_id],
                "bertscore_f1": [0.9, 0.2],
            }
        )

        aligned = align_eval_frames(model_free, bert)
        self.assertEqual(len(aligned), 2)
        self.assertAlmostEqual(safe_mean(aligned, "bertscore_f1"), 0.55, places=4)
        self.assertAlmostEqual(safe_mean(aligned, "bertscore_f1", non_abstention_only=True), 0.9, places=4)

        mismatched = bert.copy()
        mismatched["run_id"] = "other"
        self.assertTrue(align_eval_frames(model_free, mismatched).empty)


if __name__ == "__main__":
    unittest.main()
