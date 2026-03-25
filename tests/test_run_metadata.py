from __future__ import annotations

import unittest

from tests.test_support import FakeDataFrame, install_fake_pandas

install_fake_pandas()
from src.evaluation.run_metadata import annotate_with_run_metadata, compute_run_id, resolve_run_id


class TestRunMetadata(unittest.TestCase):
    def test_compute_run_id_depends_on_metadata(self) -> None:
        questions = ["q1", "q2"]
        meta_a = {"profile": "fast", "budget_seconds": 10800}
        meta_b = {"profile": "t4-safe", "budget_seconds": 10800}

        self.assertNotEqual(compute_run_id(questions, meta_a), compute_run_id(questions, meta_b))

    def test_annotate_and_resolve_run_metadata(self) -> None:
        questions = ["q1", "q2"]
        meta = {"profile": "fast", "budget_seconds": 10800}
        df = FakeDataFrame({"question": questions})

        annotated = annotate_with_run_metadata(df, questions, metadata=meta)

        self.assertIn("run_id", annotated.columns)
        self.assertIn("run_metadata_json", annotated.columns)
        self.assertIn("run_profile", annotated.columns)
        self.assertEqual(resolve_run_id(annotated, "question"), annotated["run_id"].tolist()[0])

    def test_resolve_run_id_rejects_mixed_run_ids(self) -> None:
        df = FakeDataFrame({"question": ["q1", "q2"], "run_id": ["a", "b"]})
        self.assertIsNone(resolve_run_id(df, "question"))


if __name__ == "__main__":
    unittest.main()
