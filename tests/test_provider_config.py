import os
import unittest

from pageindex.utils import ConfigLoader, _normalize_model_for_custom_base, _strip_litellm_prefix


class ProviderConfigTests(unittest.TestCase):
    def test_normalize_model_for_custom_base_adds_openai_prefix(self):
        normalized = _normalize_model_for_custom_base(
            "unsloth/MiniMax-M2.7",
            "https://openrouter.ai/api/v1",
        )
        self.assertEqual(normalized, "openai/unsloth/MiniMax-M2.7")

    def test_normalize_model_for_custom_base_keeps_provider_prefix(self):
        normalized = _normalize_model_for_custom_base(
            "openrouter/nvidia/nemotron-3-mini",
            "https://openrouter.ai/api/v1",
        )
        self.assertEqual(normalized, "openrouter/nvidia/nemotron-3-mini")

    def test_strip_litellm_prefix(self):
        self.assertEqual(
            _strip_litellm_prefix("litellm/openrouter/google/gemini-2.0-flash-001"),
            "openrouter/google/gemini-2.0-flash-001",
        )

    def test_loader_env_overrides(self):
        old = dict(os.environ)
        try:
            os.environ["PAGEINDEX_MODEL"] = "openrouter/google/gemini-2.0-flash-001"
            os.environ["PAGEINDEX_API_BASE"] = "https://openrouter.ai/api/v1"
            os.environ["PAGEINDEX_API_KEY"] = "test-key"
            cfg = ConfigLoader().load()
            self.assertEqual(cfg.model, "openrouter/google/gemini-2.0-flash-001")
            self.assertEqual(cfg.model_api_base, "https://openrouter.ai/api/v1")
            self.assertEqual(cfg.model_api_key, "test-key")
        finally:
            os.environ.clear()
            os.environ.update(old)


if __name__ == "__main__":
    unittest.main()
