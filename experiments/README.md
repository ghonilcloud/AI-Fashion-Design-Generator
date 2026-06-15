# Experiments

This folder contains the paper experiment workflow for CAD-sketch fashion generation.

The canonical pipeline is:

1. Put CAD sketch images in a dataset folder.
2. Run the full experiment grid with `run_experiment.py`.
3. Summarize `raw_results.csv` with `summarize_results.py`.

The experiment code uses the same backend modules as the app:

- Prompt generation: `backend.app.generator`
- Image generation: `backend.app.image_generator`
- Metrics: `experiments.metrics`

## Dataset

Default dataset path:

```text
datasets/cad_sketches/
```

Supported image extensions:

```text
.png, .jpg, .jpeg, .webp, .bmp
```

## Run Experiments

Smoke run on five sketches:

```bash
python experiments/run_experiment.py --dataset datasets/cad_sketches --limit 5
```

Run or rerun all sketches, replacing existing rows with the same `run_id`:

```bash
python experiments/run_experiment.py --dataset datasets/cad_sketches --overwrite
```

Optional shared descriptors can be applied to every sketch:

```bash
python experiments/run_experiment.py --dataset datasets/cad_sketches --tones Elegant,Minimalist --kansei-words Airy,Structured
```

The runner evaluates all combinations of:

- Prompt strategy: `rule_based`, `llm`
- Model key: `sd_v1_5`, `sdxl`, `ssd_1b`
- Inference steps: `10`, `20`
- Guidance scale: `7.5`
- ControlNet scale: `0.5`, `0.75`, `1.0`
- Seed: `42`

## Results Layout

```text
experiments/results/
  raw_results.csv
  images/
  prompts/
  metadata/
  summary_by_model.csv
  summary_by_prompt_strategy.csv
  summary_by_model_and_prompt.csv
  summary_by_controlnet_scale.csv
  summary_by_steps.csv
  paper_tables.md
```

`raw_results.csv` is the main experiment log. Each row is one generated output.

`images/` stores generated images.

`prompts/` stores the exact prompt text for each run.

`metadata/` stores per-run JSON metadata, including generation parameters and metric details.

## Summarize Results

After `raw_results.csv` exists:

```bash
python experiments/summarize_results.py
```

This writes summary CSV files and `paper_tables.md`, which contains markdown tables ready to paste into the paper.

## Metric Interpretation

- CLIP text-image similarity is higher-is-better.
- LPIPS perceptual distance is lower-is-better.
- Sketch IoU is higher-is-better.
- Lower latency and lower peak VRAM are better for deployment.

