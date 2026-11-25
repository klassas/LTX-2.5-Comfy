"""Configuration and defaults for the LTX-2.5 ComfyUI worker.

Mirrors the Qwen-Image / MiniMax-H3 workers' config conventions (env
names, S3 block, Pod on-demand mode) wherever the workers share a
concept, so the endpoint setup steps and the app's expectations carry
over unchanged. LTX-2.5-specific fields (model filenames, sampler
defaults) live in src/workflow.py, same split as the sibling workers.
"""

import os


def _env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


# -- Model / ComfyUI process ──────────────────────────────────────────────────
# Lowercase deliberately — see src/model_link.py's REPO_ID comment. Must match
# the literal string configured in the RunPod endpoint's Model field, not the
# Hub's mixed-case canonical id. The endpoint overrides this with the slim
# repo (klassas/ltx-2.5-slim) holding only the five needed files; the
# default is Lightricks' upstream repo (gated — needs an HF_TOKEN whose
# account accepted the LTX-2.5 license on the model page).
HF_REPO = os.getenv("HF_REPO", "lightricks/ltx-2.5")
# Where the Dockerfile installs ComfyUI itself (core only, no models baked
# in — weights come from Model Cache via src/model_link.py at runtime).
COMFY_DIR = os.getenv("COMFY_DIR", "/opt/ComfyUI")
COMFY_HOST = os.getenv("COMFY_HOST", "127.0.0.1")
COMFY_PORT = int(os.getenv("COMFY_PORT", "8188"))
# Cold start: ComfyUI's own process boot (before any model weights load —
# those load lazily per-prompt) plus this worker's model_link.py symlinking.
COMFY_BOOT_TIMEOUT_S = int(os.getenv("COMFY_BOOT_TIMEOUT_S", "120"))
# Per-job ceiling: the first prompt after boot pays weight-load time
# (~37 GB of int8 weights) on top of a two-stage sampling pass, so this
# matches the endpoint's 1800 s execution timeout.
COMFY_INFERENCE_TIMEOUT_S = int(os.getenv("COMFY_INFERENCE_TIMEOUT_S", "1800"))
# A known ComfyUI regression around pinned-memory allocation has been
# reported fixed by --disable-pinned-memory. Off by default; flip via env
# var without a rebuild if cold starts show it.
COMFY_DISABLE_PINNED_MEMORY = _env_bool("COMFY_DISABLE_PINNED_MEMORY", "0")

MODEL_TAG = os.getenv("MODEL_TAG", "ltx-2.5-comfy")


def model_tag() -> str:
    return MODEL_TAG


# -- Inference defaults ───────────────────────────────────────────────────────
# Output size (the graph generates at half size and latent-upscales x2 —
# see src/workflow.py). 1280x720 is the official LTX-2.5 template default.
DEFAULT_WIDTH = int(os.getenv("DEFAULT_WIDTH", "1280"))
DEFAULT_HEIGHT = int(os.getenv("DEFAULT_HEIGHT", "720"))
DEFAULT_SECONDS = 5
DEFAULT_FPS = 24
# Frame rates the hosted LTX-2.5 support matrix lists; 48/50 shorten the
# practical duration (see MAX_FRAMES below).
ALLOWED_FPS = (24, 25, 48, 50)
DEFAULT_SEED = -1

# Output sizes must be multiples of this. 16, not 32: the official
# templates' default output is 1280x720 and 720 is not a multiple of
# 32 — stage 1 halves the size (360) and EmptyLTXVLatentVideo floors
# it onto the VAE's 32-grid, so the template itself tolerates it.
MOD_VALUE = 16

# -- Limits ───────────────────────────────────────────────────────────────────
# Lightricks' published support matrix caps the distilled variant at
# 20 s; treat that as the ceiling (MAX_SECONDS can only lower it).
SECONDS_CEILING = 20
MAX_SECONDS = min(int(os.getenv("MAX_SECONDS", "20")), SECONDS_CEILING)
MIN_SECONDS = 1
# LTXVEmptyLatentAudio's frames_number input maxes at 1000 in ComfyUI
# 0.46.2, so the snapped frame count (seconds * fps + 1) must stay under
# it — 20 s is only reachable at 24/25 fps.
MAX_FRAMES = 1000
# The app composes long tile-based prompts (character/scene blocks), so
# the prompt ceiling is much higher than the H3 worker's 2000.
MAX_PROMPT_LENGTH = 8000
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 8192 * 8192
# Mid keyframe guides per job (the ref_images field), on top of the
# dedicated start-frame and last-frame slots. LTXVAddGuide chains with
# no hard count cap; four matches the app's reference-image plan.
MAX_REF_IMAGES = 4
# Guard rail only: two-stage 720p is the target operating point, but
# allow up to 1080p-class requests through (VRAM permitting).
MAX_RESOLUTION_AREA = int(os.getenv("MAX_RESOLUTION_AREA", str(1920 * 1080)))
MIN_DIMENSION = 128
IMAGE_DOWNLOAD_TIMEOUT = 10
UPLOAD_TIMEOUT = 60

# Fields the app sends that this worker accepts and drops: `workflow` is
# the placeholder asset the app still ships for old third-party
# endpoints — this worker builds its graph in code (src/workflow.py).
IGNORED_PARAMS = ("workflow",)

# -- S3 / object storage ──────────────────────────────────────────────────────
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "")
S3_REGION = os.getenv("S3_REGION", "auto")
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID", "")
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY", "")
S3_BUCKET_DEFAULT = os.getenv("S3_BUCKET_DEFAULT", "")
S3_PUBLIC_BASE_URL = os.getenv("S3_PUBLIC_BASE_URL", "")

# -- On-demand Pod mode (SERVE_MODE=api, see src/watchdog.py) ─────────────────
SERVE_MODE = os.getenv("SERVE_MODE", "serverless")
POD_API_PORT = os.getenv("POD_API_PORT", "8000")
POD_IDLE_TIMEOUT_S = int(os.getenv("POD_IDLE_TIMEOUT_S", "180"))
POD_MAX_LIFETIME_S = int(os.getenv("POD_MAX_LIFETIME_S", "10800"))
RUNPOD_POD_ID = os.getenv("RUNPOD_POD_ID", "")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")


# -- Extra (user) LoRAs --------------------------------------------------------
# Up to three LoRAs chained after the UNETLoader (see
# workflow._model_branch), feeding BOTH sampling stages. The files are
# NOT part of the Model Cache snapshot: at boot the worker fetches them
# itself (storage.download_extra_loras), so swapping LoRAs is an env
# change + worker restart, never a rebuild. EXTRA_LORA_KEY{,_2,_3} are
# object keys inside S3_BUCKET_DEFAULT; empty disables that slot.
# EXTRA_LORA_URL{,_2,_3} is the alternative source for a slot: the
# worker downloads the file straight from that URL at boot (for LoRAs
# too large to push into the bucket through the storage web UI). A set
# URL wins over the slot's key. LTX LoRAs are single files — no
# high/low pairing (that is Wan's design, not LTX's). Note LTX 2.3
# LoRAs do NOT work on 2.5; each endpoint carries its own set.
EXTRA_LORA_KEY = os.getenv("EXTRA_LORA_KEY", "")
EXTRA_LORA_STRENGTH = float(os.getenv("EXTRA_LORA_STRENGTH", "0.5"))
EXTRA_LORA_KEY_2 = os.getenv("EXTRA_LORA_KEY_2", "")
EXTRA_LORA_STRENGTH_2 = float(os.getenv("EXTRA_LORA_STRENGTH_2", "0.5"))
EXTRA_LORA_KEY_3 = os.getenv("EXTRA_LORA_KEY_3", "")
EXTRA_LORA_STRENGTH_3 = float(os.getenv("EXTRA_LORA_STRENGTH_3", "0.5"))
EXTRA_LORAS = [
    (EXTRA_LORA_KEY, EXTRA_LORA_STRENGTH),
    (EXTRA_LORA_KEY_2, EXTRA_LORA_STRENGTH_2),
    (EXTRA_LORA_KEY_3, EXTRA_LORA_STRENGTH_3),
]
EXTRA_LORA_URL = os.getenv("EXTRA_LORA_URL", "")
EXTRA_LORA_URL_2 = os.getenv("EXTRA_LORA_URL_2", "")
EXTRA_LORA_URL_3 = os.getenv("EXTRA_LORA_URL_3", "")
EXTRA_LORA_URLS = [EXTRA_LORA_URL, EXTRA_LORA_URL_2, EXTRA_LORA_URL_3]
