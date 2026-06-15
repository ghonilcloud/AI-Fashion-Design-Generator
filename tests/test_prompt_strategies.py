import unittest
from unittest.mock import patch

from backend.app import generator


class PromptStrategyTests(unittest.TestCase):
    def test_rule_based_prompt_is_deterministic(self):
        prompt = generator.build_rule_based_prompt(
            selected_tones=["Elegant", "Minimalist"],
            selected_kansei=["Airy", "Structured"],
            garment_type="a-line dress",
            sketch_metadata="front-view sketch",
        )

        self.assertIn("CAD-style fashion technical drawing", prompt)
        self.assertIn("a-line dress", prompt)
        self.assertIn("Elegant, Minimalist", prompt)
        self.assertIn("Airy, Structured", prompt)
        self.assertIn("front-view sketch", prompt)

    def test_rule_based_strategy_does_not_call_gemini(self):
        with patch.object(generator, "call_gemini_for_prompt") as gemini_call:
            prompt, strategy = generator.build_prompt_for_strategy(
                selected_tones=["Elegant"],
                selected_kansei=["Airy"],
                prompt_strategy="rule_based",
            )

        self.assertEqual(strategy, "rule_based")
        self.assertIn("Elegant", prompt)
        gemini_call.assert_not_called()

    def test_llm_strategy_still_uses_gemini_path(self):
        with patch.object(generator, "call_gemini_for_prompt", return_value="Gemini prompt"):
            prompt, strategy = generator.build_prompt_for_strategy(
                selected_tones=["Elegant"],
                selected_kansei=["Airy"],
                prompt_strategy="llm",
            )

        self.assertEqual(strategy, "llm")
        self.assertIn("Gemini prompt", prompt)
        self.assertIn("--rendering:", prompt)

    def test_prompt_strategy_aliases(self):
        self.assertEqual(generator.normalize_prompt_strategy("llm_prompt"), "llm")
        self.assertEqual(generator.normalize_prompt_strategy("gemini"), "llm")
        self.assertEqual(generator.normalize_prompt_strategy("rule_based_prompt"), "rule_based")


if __name__ == "__main__":
    unittest.main()

