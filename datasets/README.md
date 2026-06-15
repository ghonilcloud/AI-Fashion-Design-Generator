# Datasets

Place CAD-style fashion sketch images in `datasets/cad_sketches/`.

Use `datasets/manifests/pilot_41.csv` to describe the pilot evaluation set. Each row should provide:

```csv
sketch_id,sketch_path,garment_type,tones,kansei_words,manual_prompt
```

List fields use semicolons, for example:

```csv
sketch_001,datasets/cad_sketches/sketch_001.png,a-line dress,Elegant;Minimalist,Airy;Structured,
```
