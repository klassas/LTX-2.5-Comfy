# LTX-2.5 Worker (worker-ltx)

RunPod Serverless worker for **LTX-2.5** video generation (video +
audio in one pass) in the RunPod Media Studio app, built in the same
fkky pattern as the sibling workers (`worker-qwen/`, `worker-h3-image/`,
`worker-lora/`): a GitHub-deployed image that installs ComfyUI, links
the endpoint's **Model Cache** snapshot into ComfyUI's `models/` tree
at boot (no network volume, no monthly fee), builds the ComfyUI graph
in code from simple fields, and uploads the result MP4 to the S3
bucket, returning a presigned URL.

```
worker-ltx/
├── Dockerfile          ComfyUI 0.46.2 + torch cu130, no models baked in
├── requirements.txt    runpod / requests / websocket-client / boto3
└── src/
    ├── handler.py      RunPod entry point; boots ComfyUI as a subprocess
    ├── config.py       Env-driven config (HF_REPO, S3_*, LoRA slots, defaults)
    ├── validator.py    App-shaped input -> cleaned params
    ├── workflow.py     The LTX-2.5 two-stage graph, built per job
    ├── comfy_client.py ComfyUI HTTP/WebSocket client
    ├── model_link.py   Model Cache snapshot -> ComfyUI models/ symlinks
    ├── image.py        Start/last/guide image download + decode
    ├── storage.py      S3 upload + presigned URL, extra-LoRA download
    ├── security.py     SSRF guard for image URLs
    ├── errors.py       Shared error contract with the sibling workers
    └── watchdog.py     Pod-mode self-termination (no-op on Serverless)
```

## ⚠️ Before creating the endpoint: accept the license on Hugging Face

The LTX-2.5 model repo (`Lightricks/LTX-2.5`) is **gated**. Log in to
huggingface.co as **klassas**, open the `Lightricks/LTX-2.5` model
page, and click **"Agree and access"** once. Until that click is done,
no token can download the files and the slim-repo upload (and the
endpoint's Model Cache) cannot work. The endpoint must also carry an
`HF_TOKEN` from an account that has accepted — the klassas READ token
already used by the other endpoints works.

The license itself (LTX-2.x Community License) allows personal use
with **no territory restrictions**, so — unlike the MiniMax H3
endpoints — this endpoint needs **no data-center restriction**.

## Model set (slim repo)

The worker loads the official templates' distilled INT8 set, five
files (~39.7 GB total), mirrored in the private slim repo
**`klassas/ltx-2.5-slim`** (same folder layout as upstream):

| File | Role |
|---|---|
| `diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | transformer (distilled, INT8) |
| `text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | Gemma 4 text encoder (INT8) |
| `vae/ltx-2.5-video-vae-bf16.safetensors` | video VAE |
| `vae/ltx-2.5-audio-vae-bf16.safetensors` | audio VAE |
| `latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | ×2 latent upscaler (stage 2) |

## Endpoint settings

Create it the same way as the Qwen / H3 Image endpoints (Serverless →
"Deploy from a GitHub repository" → this repo, branch `main`,
Dockerfile path `/Dockerfile`, Queue mode):

- **GPU: 48 GB tier** (all types in the tier checked), GPU count 1,
  Max workers 1, Active workers 0
- **Cached model:** `klassas/ltx-2.5-slim`
- **Container disk:** 20 GB (weights live in Model Cache, not the disk)
- **Idle timeout:** 60 s · **Initialization timeout:** 420 s
- **Execution timeout:** enabled, **1800** s
- **Data centers:** no restriction needed (see license note above)
- **Environment variables** (Edit endpoint → Raw editor, after
  creation — the create form does not reliably persist them):

```
S3_ENDPOINT=...
S3_REGION=...
S3_ACCESS_KEY_ID=...
S3_SECRET_ACCESS_KEY=...
S3_BUCKET_DEFAULT=...
HF_REPO=klassas/ltx-2.5-slim
HF_TOKEN=...        (the klassas READ token; required — gated source)
```

Optional LoRA slots (see below): `EXTRA_LORA_KEY` / `EXTRA_LORA_URL`
(+ `_2` / `_3`) and `EXTRA_LORA_STRENGTH` (+ `_2` / `_3`).

After the endpoint is created and shows Ready, enter the new API key
(if changed) and this endpoint's ID in the app's Settings tab, then run
one short test generation (5 s, no images) and keep the app open while
it runs.

## Input contract

Flat fields; the graph (ComfyUI's official two-stage LTX-2.5 template
shape) is built in code by `src/workflow.py`. The prompt passes
through as typed — there is no prompt-enhancer stage.

| Field | Meaning |
|---|---|
| `prompt` (required) | Positive prompt |
| `negative_prompt` | Negative prompt |
| `seconds` | 1–20, default 5. Frames = seconds × fps + 1, snapped down to the 1 + 8k grid |
| `fps` | 24 (default) / 25 / 48 / 50. 48/50 fps caps duration at ~10 s (1000-frame audio-latent limit) |
| `width` / `height` | Output size, both or neither, multiples of 16, default 1280×720. Stage 1 generates at half size, the latent upscaler ×2 restores it |
| `seed` | Stage-1 seed (-1 = random). Stage 2's seed is fixed at 42, as in the official templates |
| `image_base64` / `image_url` | Start frame — LTXVImgToVideoInplace in both stages (strength 0.7, then 1.0) |
| `last_image_base64` / `last_image_url` | Last frame — LTXVAddGuide at frame_idx −1 |
| `ref_images` | Up to 4 mid keyframe guides — LTXVAddGuide at evenly spread frame positions, strength 0.7. Items: `{"base64": ...}` / `{"url": ...}` / plain base64 string |
| `lora_strength` / `lora_strength_2` / `lora_strength_3` | 0–2 per-request strengths for the env-configured LoRA slots; 0 disables that slot for the job |
| `mute` | true = return the MP4 without the audio track |
| `output` | `{"bucket", "prefix", "public": false, "ttl_seconds"}` — same B2 presigned-URL block as the other workers (default prefix `ltx-2.5/`) |

Response: `{"status": "completed", "video_url": "<presigned B2 URL>",
"meta": {...}, "timing": {...}}` — the same `video_url` key the H3
worker returns, so the app's output handling is unchanged.

Modes are implicit: no images = text-to-video; start frame =
image-to-video; start + last = first/last-frame; `ref_images` alone =
guided generation. Audio (dialogue, ambience, effects) is generated
jointly with the video in the same diffusion pass — that is LTX-2.5's
defining feature — unless `mute` is set.

## LoRA

Standard single-file LoRAs via `LoraLoaderModelOnly`, chained after
the UNETLoader and feeding **both** sampling stages (per Lightricks'
two-stage guide). No high/low pairing (that is Wan's design). The
three slots are configured at deploy time, exactly like the H3 worker:

- `EXTRA_LORA_KEY{,_2,_3}` — object key inside the S3 bucket, or
- `EXTRA_LORA_URL{,_2,_3}` — direct download URL (wins over the key;
  for files too large for the storage web UI's 500 MB upload limit).

The worker downloads them at boot into ComfyUI's `models/loras/` and
**fails the boot if a configured LoRA cannot be fetched**. Changing
LoRAs = edit the env vars + restart the worker; no rebuild.

⚠️ **LTX 2.3 LoRAs do not work on 2.5.** aki's memo LoRAs belong to
the future LTX 2.3 endpoint; this endpoint needs 2.5 LoRAs.

## Graph notes (choices made where the templates diverge)

The T2V/I2V templates are two-stage; the FLF template is single-stage.
This worker unifies all modes into the two-stage graph:

- Guides are applied to the stage-1 (half-size) video latent and the
  appended guide frames are removed with **LTXVCropGuides after
  stage 1, before the latent upscaler**; stage 2's guider reads
  CropGuides' cleaned conditioning. (In single-stage FLF the crop sits
  right before decoding — same intent, moved to the stage boundary.)
- With guides present, stage 1's sampler output is read from
  `denoised_output` (as FLF does); otherwise from `output` (as
  T2V/I2V do).
- The start frame is re-applied after the upscaler with
  LTXVImgToVideoInplace at strength 1.0 (the I2V template's stage-2
  node), on the upscaled (and, with guides, cropped) latent.
- The audio VAE is loaded with a plain **VAELoader**, exactly as the
  official templates wire it (the file lives in `vae/`).
  `LTXVAudioVAELoader` is not used — it expects the file under
  `checkpoints/`.
- Sampler schedules are the templates' distilled ones: stage 1
  ManualSigmas `1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375,
  0.725, 0.421875, 0.0`, stage 2 `0.85, 0.7250, 0.4219, 0.0`, both
  `euler_ancestral` with LTXVDualCFGGuider cfg [1, 1].

## Local checks

```
python3 -m py_compile src/*.py
```

The Dockerfile also runs a build-time graph/validator check (a T2V
graph, an FLF graph with guides + crop wiring, a LoRA chain, and
validator cases) so a graph-construction typo fails the image build
instead of the first job.
