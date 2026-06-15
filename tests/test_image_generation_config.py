import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

try:
    from backend.app import image_generator
except Exception as exc:
    image_generator = None
    IMAGE_GENERATOR_IMPORT_ERROR = exc
else:
    IMAGE_GENERATOR_IMPORT_ERROR = None


class _FakePipeline:
    def __call__(self, **kwargs):
        return type("Result", (), {"images": [Image.new("RGB", (8, 8), "white")]})()


class ImageGenerationConfigTests(unittest.TestCase):
    def setUp(self):
        if image_generator is None:
            self.skipTest(f"image_generator import failed: {IMAGE_GENERATOR_IMPORT_ERROR}")

    def test_model_registry_resolves_simple_keys(self):
        config = image_generator.resolve_model_config("sd_v1_5")

        self.assertEqual(config["model_key"], "sd_v1_5")
        self.assertEqual(config["base_model"], "runwayml/stable-diffusion-v1-5")
        self.assertEqual(config["controlnet_model"], "lllyasviel/sd-controlnet-canny")

    def test_generation_filename_contains_reproducibility_fields(self):
        filename = image_generator.build_generation_filename(
            sketch_id="sketch_001",
            model_key="sd_v1_5",
            prompt_strategy="rule_based",
            num_inference_steps=20,
            guidance_scale=7.5,
            controlnet_conditioning_scale=1.0,
            seed=42,
        )

        self.assertEqual(
            filename,
            "sketch_001_sd_v1_5_rule_based_steps20_cfg7p5_cnet1p0_seed42.png",
        )

    def test_generate_returns_metadata_and_uses_encoded_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sketch_path = Path(tmpdir) / "sketch.png"
            Image.new("RGB", (8, 8), "black").save(sketch_path)

            with patch.object(image_generator, "load_pipeline", return_value=_FakePipeline()):
                metadata = image_generator.generate_fashion_design(
                    sketch_path=sketch_path,
                    prompt="test prompt",
                    output_path=tmpdir,
                    model_key="sd_v1_5",
                    num_inference_steps=10,
                    guidance_scale=6.5,
                    controlnet_conditioning_scale=0.75,
                    seed=123,
                    prompt_strategy="llm",
                    sketch_id="case_a",
                    device="cpu",
                    dtype="float32",
                )

            output_path = Path(metadata["output_path"])
            self.assertTrue(output_path.exists())
            self.assertEqual(metadata["model_key"], "sd_v1_5")
            self.assertEqual(metadata["num_inference_steps"], 10)
            self.assertEqual(metadata["guidance_scale"], 6.5)
            self.assertEqual(metadata["controlnet_conditioning_scale"], 0.75)
            self.assertEqual(metadata["seed"], 123)
            self.assertEqual(
                output_path.name,
                "case_a_sd_v1_5_llm_steps10_cfg6p5_cnet0p75_seed123.png",
            )


if __name__ == "__main__":
    unittest.main()
