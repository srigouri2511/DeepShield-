import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import api


class OpenEnvApiTests(unittest.TestCase):
    def setUp(self):
        self.client = api.app.test_client()

    def test_health_endpoint_reports_openenv_ready(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["openenv"])

    def test_reset_accepts_difficulty_and_returns_observation(self):
        response = self.client.post("/reset?difficulty=easy")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("observation", payload)
        self.assertEqual(payload["observation"]["task"], "phishing_triage")
        self.assertEqual(payload["observation"]["step"], 0)

    def test_step_and_state_return_openenv_payloads(self):
        self.client.post("/reset", json={"task": "phishing_triage"})
        response = self.client.post("/step", json={
            "task": "phishing_triage",
            "decision": "phishing",
            "confidence": 0.8,
            "reasoning": "test",
        })
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("observation", payload)
        self.assertIn("reward", payload)
        self.assertIn("done", payload)
        self.assertIn("info", payload)
        self.assertIn("value", payload["reward"])

        state_response = self.client.get("/state")
        self.assertEqual(state_response.status_code, 200)
        state_payload = state_response.get_json()
        self.assertIn("state", state_payload)
        self.assertEqual(state_payload["state"]["active_task"], "phishing_triage")


if __name__ == "__main__":
    unittest.main()
