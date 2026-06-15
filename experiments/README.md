# Experiments

This folder contains a reproducible experiment scaffold that wraps the existing
application code:

- Gemini prompt generation uses `backend.app.generator`.
- Image generation uses `backend.app.image_generator`.
- Results are written under `experiments/results/`.

## Dataset Manifest

Add sketch rows to `datasets/manifests/pilot_41.csv`:

```csv
sketch_id,sketch_path,garment_type,tones,kansei_words,manual_prompt
sketch_001,datasets/sketches/sketch_001.png,a-line dress,Elegant;Minimalist,Airy;Structured,
```

Use semicolons for multiple tones or Kansei words.

## Prompt-Only Smoke Tests

Prompt-only mode avoids diffusion model loading:

```bash
python -m experiments.runners.run_single --sketch datasets/sketches/sketch_001.png --prompt-strategy rule_based --tones Elegant --kansei-words Airy --prompt-only
```

```bash
python -m experiments.runners.run_batch --config experiments/configs/sd15.json --manifest datasets/manifests/pilot_41.csv --prompt-only
```

## Full Generation

```bash
python -m experiments.runners.run_batch --config experiments/configs/sd15.json --manifest datasets/manifests/pilot_41.csv
```

## Comparisons

```bash
python -m experiments.runners.run_prompt_comparison --manifest datasets/manifests/pilot_41.csv
```

```bash
python -m experiments.runners.run_ablation --manifest datasets/manifests/pilot_41.csv
```

```bash
python -m experiments.runners.run_model_comparison --manifest datasets/manifests/pilot_41.csv
```

The SDXL and SSD-1B configs document the intended paper comparison, but the
current backend still uses a Stable Diffusion ControlNet pipeline class. Those
configs may require a dedicated SDXL-compatible pipeline implementation before
they run successfully.

## Metrics

Currently runnable with installed dependencies:

- `sketch_iou`
- latency and optional CUDA peak memory logging

Optional modules are scaffolded for:

- `clip`
- `lpips`

Those may require model-weight downloads or additional packages before use.
