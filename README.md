# AI Fashion Design Generator
Turn a fashion sketch into a polished concept render with Gemini-shaped prompts and Stable Diffusion + ControlNet.

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](#prerequisites--installation)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)](#about--features)
[![Docker](https://img.shields.io/badge/Docker-Supported-2496ED?logo=docker&logoColor=white)](#docker)
[![License](https://img.shields.io/badge/License-Unspecified-lightgrey)](#license)

## About / Features
AI Fashion Design Generator is a sketch-to-fashion workflow for designers, students, and concept artists who want fast visual iteration without losing the original silhouette.

What it does:
- Upload a CAD-style sketch and generate a styled fashion render.
- Choose from curated tone labels and Kansei words to steer the result.
- Return both the generated image and the exact prompt used to create it.
- Serve the frontend and backend from one FastAPI app for simple local runs.

Why use it:
- It preserves structure instead of only generating from text.
- It keeps the creative direction controlled through a small, predictable vocabulary.
- It shows the prompt behind the output, which makes iteration faster and easier to debug.

How the AI works:
- This repo does not train a custom model from scratch.
- Gemini 2.5 Flash is used as the prompt-shaping layer. It reads the uploaded sketch plus the selected tones/Kansei words and expands them into a compact fashion prompt.
- Stable Diffusion v1.5 + ControlNet does the image synthesis step, using the sketch as structural guidance so the silhouette stays intact.
- The current generation settings are fixed in code for consistency:
  - `num_inference_steps=20`
  - `guidance_scale=7.5`
  - `controlnet_conditioning_scale=1.0`
  - Negative prompt: `blurry, low quality, distorted, ugly, bad anatomy`
- The app preprocesses sketches to a 512 x 512 grayscale input and falls back to returning the sketch if image generation dependencies are unavailable.

Built-in creative vocabulary:
- Tones: Elegant, Minimalist, Avant-garde, Street, Romantic, Futuristic, Technical, Playful, Luxurious, Sustainable
- Kansei words: Airy, Structured, Fluid, Geometric, Organic, Textured, Matte, Sheer, Layered, Tailored, Architectural, Ergonomic, Technical, Deconstructed, Streamlined, Ornamental, Monochrome, Vibrant, Contrast, Soft

## Prerequisites
- Python 3.11 or newer
- Git
- A Gemini API key
- A Hugging Face token is recommended for model downloads
- An NVIDIA GPU is recommended for best results, but the app can still boot without one
- Docker is optional

## Prerequisites & Installation

### 1. Clone the repository
```bash
git clone https://github.com/ghonilcloud/AIFashionChatBot.git
cd AIFashionChatBot
```

### 2. Create a virtual environment
```bash
python -m venv .venv
```

### 3. Activate it
```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Add environment variables
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
HUGGINGFACE_TOKEN=your_huggingface_token_here
```

Notes:
- `GEMINI_API_KEY` is required.
- `HUGGINGFACE_TOKEN` is used when downloading the Stable Diffusion and ControlNet weights.
- The backend also accepts `API_KEY` as a fallback, but `GEMINI_API_KEY` is the preferred name.

### 6. Start the app
```bash
python start_server.py
```

The app starts on `http://localhost:8000`.

### Docker
```bash
docker build -t aifashionchatbot .
docker run -p 8000:8000 --gpus all aifashionchatbot
```

## Usage

### Local web UI
1. Start the server.
2. Open `http://localhost:8000`.
3. Upload a sketch image.
4. Select up to 3 tones and any Kansei words.
5. Click **Generate Design**.

### API checks
```bash
curl http://localhost:8000/health
```

```bash
curl http://localhost:8000/api/tones
```

### Local frontend note
The frontend template currently includes a production API override in `frontend/index.html`.
For local development, point `window.API_BASE_URL` at `http://localhost:8000`, or remove that override so `frontend/script.js` can use its localhost default.

## Contributing
Issues and pull requests are welcome.

- If you find a bug or want a feature, open an issue with the expected behavior and a clear repro.
- If you want to contribute code, fork the repo, make the change, and submit a pull request.
- Feel free to copy, adapt, and use the project for your own experiments.

## License
No license file is included in this repository.

If you plan to redistribute or reuse the project publicly, add a `LICENSE` file first and reference it here.
