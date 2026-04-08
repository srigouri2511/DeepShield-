import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import inference
from env.models import TaskName


class _DummyMessage:
    def __init__(self, content: str):
        self.content = content


class _DummyChoice:
    def __init__(self, content: str):
        self.message = _DummyMessage(content)


class _DummyResponse:
    def __init__(self, content: str):
        self.choices = [_DummyChoice(content)]


class _DummyCompletions:
    def create(self, **kwargs):
        return _DummyResponse('{"decision":"legitimate","confidence":0.51,"reasoning":"stub"}')


class _DummyChat:
    def __init__(self):
        self.completions = _DummyCompletions()


class _DummyClient:
    def __init__(self):
        self.chat = _DummyChat()


class InferenceTests(unittest.TestCase):
    def test_create_client_requires_hf_token(self):
        original = inference.HF_TOKEN
        original_env = inference.os.environ.get("HF_TOKEN")
        inference.HF_TOKEN = None
        try:
            inference.os.environ.pop("HF_TOKEN", None)
            with self.assertRaises(ValueError):
                inference.create_client()
        finally:
            inference.HF_TOKEN = original
            if original_env is None:
                inference.os.environ.pop("HF_TOKEN", None)
            else:
                inference.os.environ["HF_TOKEN"] = original_env

    def test_resolve_client_type_prefers_hybrid_mapping(self):
        clients = {"hf": _DummyClient(), "openai": _DummyClient()}
        provider = inference.resolve_client_type(TaskName.IDENTITY_FRAUD, "hybrid", clients)
        self.assertEqual(provider, "openai")

    def test_run_episode_emits_required_output_lines(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            inference.run_episode(
                {"openai": _DummyClient()},
                {"openai": "demo-model", "hf": "hf-demo-model"},
                "openai",
                TaskName.PHISHING_TRIAGE,
            )

        lines = [line for line in buffer.getvalue().splitlines() if line.strip()]
        self.assertGreaterEqual(len(lines), 3)
        self.assertTrue(
            lines[0].startswith(
                "[START] task=phishing_triage env=deepfake-phishing-detection model=demo-model"
            )
        )
        self.assertTrue(all(line.startswith("[STEP]  ") for line in lines[1:-1]))
        self.assertTrue(lines[-1].startswith("[END]   success="))
        self.assertTrue(all(" reward=" in line and " done=" in line and " error=" in line for line in lines[1:-1]))
        self.assertIn(" score=", lines[-1])
        self.assertIn(" accuracy=", lines[-1])
        self.assertIn(" avg_reward=", lines[-1])
        self.assertIn(" calibration=", lines[-1])

    def test_summarize_suite_returns_aggregate_baseline(self):
        summary = inference.summarize_suite([
            {"task": "phishing_triage", "success": True, "score": 0.9},
            {"task": "identity_fraud", "success": False, "score": 0.7},
        ])

        self.assertFalse(summary["success"])
        self.assertEqual(summary["tasks"], 2)
        self.assertEqual(summary["baseline_score"], 0.8)
        self.assertEqual(summary["task_scores"], "phishing_triage:0.9000,identity_fraud:0.7000")


if __name__ == "__main__":
    unittest.main()
