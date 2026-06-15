import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from experiments.metrics.evaluator import evaluate_generation
from experiments.metrics.sketch_iou import compute_sketch_iou


class MetricsEvaluatorTests(unittest.TestCase):
    def test_sketch_iou_uses_configurable_canny_thresholds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = Path(tmpdir) / "first.png"
            second = Path(tmpdir) / "second.png"
            _draw_box(first)
            _draw_box(second)

            score = compute_sketch_iou(
                sketch_path=first,
                generated_path=second,
                canny_low_threshold=50,
                canny_high_threshold=150,
                size=(64, 64),
            )

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_evaluate_generation_returns_expected_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sketch = Path(tmpdir) / "sketch.png"
            generated = Path(tmpdir) / "generated.png"
            _draw_box(sketch)
            _draw_box(generated)

            results = evaluate_generation(
                sketch_path=sketch,
                generated_image_path=generated,
                prompt="a structured minimalist dress",
                metrics_config={
                    "compute_clip": False,
                    "compute_lpips": False,
                    "compute_sketch_iou": True,
                    "canny_low_threshold": 50,
                    "canny_high_threshold": 150,
                    "latency_seconds": 1.25,
                    "peak_vram_gb": None,
                },
            )

        self.assertIn("clip_score", results)
        self.assertIn("lpips_score", results)
        self.assertIn("sketch_iou", results)
        self.assertEqual(results["canny_low_threshold"], 50)
        self.assertEqual(results["canny_high_threshold"], 150)
        self.assertEqual(results["clip_model_name"], "openai/clip-vit-base-patch32")
        self.assertEqual(results["lpips_backbone"], "alex")
        self.assertEqual(results["latency_seconds"], 1.25)
        self.assertIsNone(results["peak_vram_gb"])
        self.assertIsInstance(results["sketch_iou"], float)


def _draw_box(path: Path) -> None:
    image = Image.new("RGB", (64, 64), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((16, 12, 48, 52), outline="black", width=2)
    image.save(path)


if __name__ == "__main__":
    unittest.main()

