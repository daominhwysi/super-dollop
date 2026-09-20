import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

from sequence_labelling.api import app


class TestAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health(self):
        response = self.client.get("/v1/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "service": "sequence-labelling-worker"})

    @patch("sequence_labelling.api.chat")
    def test_chat_completions_endpoint(self, mock_chat):
        mock_chat.return_value = "Mocked chat response from agy"

        payload = {
            "prompt": "Hello world",
            "model": "gemini-3.8-flash-high",
            "provider": "agy"
        }
        response = self.client.post("/v1/chat/completions", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["provider"], "agy")
        self.assertEqual(data["choices"][0]["message"]["content"], "Mocked chat response from agy")

    @patch("sequence_labelling.api.asyncio.to_thread")
    def test_answer_keys_map_uses_provider(self, mock_to_thread):
        mock_to_thread.return_value = '{"Q1": "A", "Q2": "B"}'

        response = self.client.post(
            "/v1/answer-keys/map",
            data={"raw_text": "1. A 2. B", "questions": "[]"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["mapping"], {"Q1": "A", "Q2": "B"})


if __name__ == "__main__":
    unittest.main()
