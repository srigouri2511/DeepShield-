import unittest
from pathlib import Path

import api


VALID_DECISIONS = {
    "phishing_triage": {"phishing", "legitimate"},
    "url_reputation": {"suspicious", "clean"},
    "email_header_analysis": {"suspicious", "clean"},
    "deepfake_detection": {"deepfake", "real"},
    "identity_fraud": {"fraud", "legitimate"},
}


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = api.app.test_client()
        self.original_save_history = api.save_history
        api.save_history = lambda entry: None

    def tearDown(self):
        api.save_history = self.original_save_history

    def test_detect_email_uses_live_input_and_frontend_alias(self):
        response = self.client.post(
            "/api/detect",
            json={"type": "email", "input": "Urgent: verify your account now and click here.", "use_ai": False},
        )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["task"], "phishing_triage")
        self.assertEqual(data["label"], "phishing")
        self.assertEqual(data["input"], "Urgent: verify your account now and click here.")
        self.assertEqual(data["analysis_mode"], "heuristic")

    def test_detect_image_uses_requested_path(self):
        sample_path = Path(__file__).resolve().parents[1] / "samples" / "real_sample.png"
        response = self.client.post(
            "/api/detect",
            json={"task": "deepfake_detection", "input": str(sample_path), "use_ai": False},
        )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["task"], "deepfake_detection")
        self.assertEqual(data["label"], "real")
        self.assertEqual(data["analysis_mode"], "heuristic")

    def test_episode_runner_uses_valid_labels_for_every_task(self):
        for task, valid_decisions in VALID_DECISIONS.items():
            response = self.client.post("/api/episode", json={"task": task})
            self.assertEqual(response.status_code, 200, task)
            data = response.get_json()
            self.assertEqual(data["task"], task)
            self.assertGreater(len(data["steps"]), 0)
            self.assertTrue(all(step["decision"] in valid_decisions for step in data["steps"]), task)

    def test_status_route_exposes_ai_capability(self):
        response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("ai_enabled", data)
        self.assertIn("model", data)
        self.assertIn("default_analysis_mode", data)


if __name__ == "__main__":
    unittest.main()
