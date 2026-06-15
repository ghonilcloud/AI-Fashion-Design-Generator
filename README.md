# AI Fashion Design Generator

AI Fashion Design Generator is a research prototype for emotionally descriptive CAD-style fashion generation. It combines Kansei descriptors, LLM-based prompt generation, deterministic rule-based prompt baselines, and sketch-guided diffusion models to turn flat garment sketches into styled fashion concept renders.

The project supports both an interactive FastAPI/frontend demo and a reproducible experiment workflow for paper results.

## Project Overview

The pipeline is:

1. A user or experiment runner provides a CAD-style fashion sketch.
2. The design intent is described through tone labels and Kansei descriptors.
3. A prompt is generated using either:
   - `llm`: Gemini expands the descriptors and sketch context into an image prompt.
   - `rule_based`: a deterministic template creates a non-LLM baseline prompt.
4. Stable Diffusion + ControlNet generates a sketch-guided fashion image.
5. Experiments evaluate CLIP similarity, LPIPS distance, Sketch IoU, latency, and peak VRAM.

Supported model keys:

- `sd_v1_5`
- `sdxl`
- `ssd_1b`

## Setup

Use Python 3.11 or newer.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

Required environment variables:

```env
GEMINI_API_KEY=your_gemini_api_key_here
HUGGINGFACE_TOKEN=your_huggingface_token_here
```

`GEMINI_API_KEY` is required for LLM prompt generation. `HUGGINGFACE_TOKEN` is used for downloading diffusion and ControlNet model weights when needed.

## Dataset Format

Place CAD sketch images in a folder such as:

```text
datasets/cad_sketches/
  sketch_001.png
  sketch_002.png
```

Supported image extensions for experiments:

```text
.png, .jpg, .jpeg, .webp, .bmp
```

## Running A Single Generation

### CLI

Run the interactive CLI:

```bash
python tools/generate_design.py datasets/cad_sketches/sketch_001.png
```

The CLI asks for tones, Kansei words, and prompt strategy, then generates an image using the default generation settings.

Optional generation settings can be passed as flags:

```bash
python tools/generate_design.py datasets/cad_sketches/sketch_001.png --model-key sdxl --num-inference-steps 20 --guidance-scale 7.5 --controlnet-conditioning-scale 0.75 --seed 42
```

### Web App

Start the backend and frontend:

```bash
python start_server.py
```

Open:

```text
http://localhost:8000
```

The frontend supports simple generation by default and optional Advanced Settings for prompt strategy, model, inference steps, CFG scale, ControlNet scale, and seed.

### API

Health check:

```bash
curl http://localhost:8000/health
```

Generate one design:

```bash
curl -X POST http://localhost:8000/api/generate-design \
  -F "image=@datasets/cad_sketches/sketch_001.png" \
  -F "kansei_text=Fashion design based on selected tones and Kansei words" \
  -F "tones=Elegant" \
  -F "kansei_words=Airy" \
  -F "prompt_strategy=llm" \
  -F "model_key=sd_v1_5" \
  -F "num_inference_steps=20" \
  -F "guidance_scale=7.5" \
  -F "controlnet_conditioning_scale=1.0" \
  -F "seed=42"
```

## Running Full Experiments

Run the full experiment grid on a small subset:

```bash
python experiments/run_experiment.py --dataset datasets/cad_sketches --limit 5
```

Run all sketches and overwrite existing rows with the same `run_id`:

```bash
python experiments/run_experiment.py --dataset datasets/cad_sketches --overwrite
```

The experiment grid covers:

- Prompt strategy: `rule_based`, `llm`
- Model key: `sd_v1_5`, `sdxl`, `ssd_1b`
- Inference steps: `10`, `20`
- Guidance scale: `7.5`
- ControlNet scale: `0.5`, `0.75`, `1.0`
- Seed: `42`

## Summarizing Results

After `experiments/results/raw_results.csv` exists, run:

```bash
python experiments/summarize_results.py
```

This creates grouped CSV summaries and markdown tables for the paper.

## Running Statistical Tests

Run paired statistical tests comparing `rule_based` and `llm` prompt strategies:

```bash
python experiments/statistical_tests.py
```

Pairs are matched by sketch, model, inference steps, CFG scale, ControlNet scale, and seed. The script tests normality of paired differences, then uses either a paired t-test or Wilcoxon signed-rank test.

## Output Files

Main experiment outputs are written under:

```text
experiments/results/
```

Important files:

- `raw_results.csv`: one row per generated experiment output, including run ID, sketch ID, prompt strategy, model key, hyperparameters, prompt, image path, metrics, latency, VRAM, and timestamp.
- `summary_by_model.csv`: grouped mean/std results by model.
- `summary_by_prompt_strategy.csv`: grouped mean/std results by prompt strategy.
- `summary_by_model_and_prompt.csv`: grouped results by model and prompt strategy.
- `summary_by_controlnet_scale.csv`: ControlNet scale ablation summary.
- `summary_by_steps.csv`: inference step ablation summary.
- `paper_tables.md`: markdown tables ready to paste into the paper.
- `statistical_tests.csv`: machine-readable paired statistical test results.
- `statistical_tests.md`: paper-readable statistical test explanations and interpretations.

Other result folders:

- `images/`: generated images.
- `prompts/`: exact prompt text for each run.
- `metadata/`: per-run JSON metadata.

## Reproducibility Notes

The research runner is designed for reproducible experiments:

- A fixed seed is used by default: `42`.
- Model IDs are logged through `model_key`, `base_model`, and `controlnet_model`.
- Hyperparameters are logged, including inference steps, CFG scale, ControlNet scale, negative prompt, device, and dtype.
- Prompt strategy is logged as `llm` or `rule_based`.
- Metrics configuration is recorded in per-run metadata, including CLIP model name, LPIPS backbone, and Canny thresholds for Sketch IoU.
- Output filenames encode sketch ID, model key, prompt strategy, steps, CFG scale, ControlNet scale, and seed.

Model weights are not committed to this repository. Diffusers downloads them from Hugging Face using the logged model IDs and the local Hugging Face cache. For reproducible paper runs, pin the model IDs in `backend/app/image_generator.py`, keep `HUGGINGFACE_TOKEN` in `.env`, and record the generated metadata rather than committing multi-GB checkpoint files.

Metric interpretation:

- CLIP text-image similarity: higher is better.
- LPIPS perceptual distance: lower is better.
- Sketch IoU: higher is better.
- Latency and peak VRAM: lower is better for deployment.

## Docker

Docker support is included for deployment-oriented runs:

```bash
docker build -t ai-fashion-design-generator .
docker run -p 8000:8000 --gpus all ai-fashion-design-generator
```

## License

No license file is currently included. Add a `LICENSE` file before public redistribution or reuse.
