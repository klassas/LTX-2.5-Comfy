"""Input validation for the LTX-2.5 ComfyUI worker.

The field set is what the RunPod Media Studio app sends for video
models (see app RunPodClient.kt): flat fields prompt / negative_prompt
/ image_url / image_base64 / width / height, plus a `workflow` object
this worker accepts and ignores — it builds the graph itself in
src/workflow.py, same contract as the sibling workers.

LTX-2.5's image inputs are three distinct slots, NOT mutually
exclusive:

* `image_base64` / `image_url` — the start frame
  (LTXVImgToVideoInplace, both sampling stages).
* `last_image_base64` / `last_image_url` — the last frame
  (LTXVAddGuide at frame_idx=-1).
* `ref_images` — up to config.MAX_REF_IMAGES mid keyframe guides
  (LTXVAddGuide at spread frame positions). Items are
  {"base64": ...} / {"url": ...} objects, or a plain string (base64),
  the same shape the app already sends to the H3 R2V worker.

Frame count: the templates compute length = seconds * fps + 1 and
EmptyLTXVLatentVideo floors it onto the 1 + 8k latent grid internally
(((length - 1) // 8) + 1 latent frames); the validator snaps it down
to that same grid up front so every consumer (audio latent, guide
positions) sees the real frame count.
"""

from src import config
from src.errors import ValidationError


def _validate_image_source(inp: dict, url_key: str, b64_key: str) -> dict | None:
    """Normalize one image slot to {"url": ...} / {"base64": ...} / None."""
    has_url = bool(inp.get(url_key))
    has_b64 = bool(inp.get(b64_key))
    if has_url and has_b64:
        raise ValidationError(
            f"Provide either {url_key} or {b64_key}, not both.",
            hint="Remove one of the image source fields.",
        )
    if has_url:
        return {"url": inp[url_key]}
    if has_b64:
        return {"base64": inp[b64_key]}
    return None


def _validate_ref_images(inp: dict) -> list:
    """Normalize `ref_images` to a list of {"url": ...} / {"base64": ...}.

    Empty list when the field is absent. Each item is an object with
    exactly one of url / base64, or a plain string (base64 / data URL).
    """
    raw = inp.get("ref_images")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValidationError(
            "ref_images must be a list.",
            hint='Send a list like [{"base64": "..."}, {"url": "..."}].',
        )
    if len(raw) > config.MAX_REF_IMAGES:
        raise ValidationError(
            f"ref_images accepts at most {config.MAX_REF_IMAGES} images.",
            hint=f"You sent {len(raw)}.",
        )
    out = []
    for i, item in enumerate(raw):
        if isinstance(item, str):
            if item.strip():
                out.append({"base64": item})
            continue
        if not isinstance(item, dict):
            raise ValidationError(
                f"ref_images[{i}] must be an object or a base64 string.",
            )
        url, b64 = item.get("url"), item.get("base64")
        if bool(url) == bool(b64):
            raise ValidationError(
                f"ref_images[{i}] needs exactly one of 'url' or 'base64'.",
            )
        if url is not None and not isinstance(url, str):
            raise ValidationError(f"ref_images[{i}].url must be a string.")
        if b64 is not None and not isinstance(b64, str):
            raise ValidationError(f"ref_images[{i}].base64 must be a string.")
        out.append({"url": url} if url else {"base64": b64})
    return out


def _strength(inp: dict, field: str, default: float) -> float:
    """Per-request LoRA strength (ported from the H3 worker). The LoRA
    files themselves are deploy-time choices (config.EXTRA_LORA_*);
    per request only their strengths are tunable, 0 disabling that
    slot for the job."""
    v = inp.get(field, default)
    if not isinstance(v, (int, float)) or isinstance(v, bool) \
            or v < 0 or v > 2:
        raise ValidationError(
            f"{field} must be a number between 0 and 2.",
            hint=f"You sent {field}={v!r}.",
        )
    return float(v)


def validate_input(inp: dict) -> dict:
    """Validate and normalize job input. Returns a cleaned dict with defaults applied."""
    if not isinstance(inp, dict):
        raise ValidationError(
            "Input must be a JSON object.",
            hint="Send a JSON object with at least a 'prompt' field.",
        )

    start_image = _validate_image_source(inp, "image_url", "image_base64")
    last_image = _validate_image_source(inp, "last_image_url", "last_image_base64")
    ref_images = _validate_ref_images(inp)

    prompt = inp.get("prompt", "")
    if not isinstance(prompt, str):
        raise ValidationError("prompt must be a string.")
    prompt = prompt.strip()
    if not prompt:
        raise ValidationError(
            "prompt is required.",
            hint="Provide a text description of the desired video.",
        )
    if len(prompt) > config.MAX_PROMPT_LENGTH:
        raise ValidationError(
            f"prompt exceeds max length ({config.MAX_PROMPT_LENGTH} chars).",
            hint="Shorten your prompt.",
        )

    negative_prompt = inp.get("negative_prompt", "")
    if not isinstance(negative_prompt, str):
        raise ValidationError("negative_prompt must be a string.")
    if len(negative_prompt) > config.MAX_PROMPT_LENGTH:
        raise ValidationError(
            f"negative_prompt exceeds max length ({config.MAX_PROMPT_LENGTH} chars).",
        )

    seconds = inp.get("seconds", config.DEFAULT_SECONDS)
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) \
            or seconds < config.MIN_SECONDS or seconds > config.MAX_SECONDS:
        raise ValidationError(
            f"seconds must be a number between {config.MIN_SECONDS} and {config.MAX_SECONDS}.",
            hint=f"You sent seconds={seconds!r}.",
        )

    fps = inp.get("fps", config.DEFAULT_FPS)
    if not isinstance(fps, int) or isinstance(fps, bool) \
            or fps not in config.ALLOWED_FPS:
        raise ValidationError(
            f"fps must be one of {list(config.ALLOWED_FPS)}.",
            hint=f"You sent fps={fps!r}.",
        )

    # Snap onto the 1 + 8k latent grid (see module docstring).
    frames = int(round(seconds * fps)) + 1
    frames = 1 + ((frames - 1) // 8) * 8
    if frames > config.MAX_FRAMES:
        raise ValidationError(
            f"seconds={seconds} at fps={fps} needs {frames} frames, over the "
            f"{config.MAX_FRAMES}-frame limit.",
            hint="Lower the seconds or the fps (48/50 fps caps at about 10 s).",
        )

    seed = inp.get("seed", config.DEFAULT_SEED)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValidationError("seed must be an integer.")

    mute = inp.get("mute", False)
    if not isinstance(mute, bool):
        raise ValidationError("mute must be a boolean.")

    # Size: both-or-neither. These are the OUTPUT dimensions; the graph
    # generates at half size and latent-upscales x2.
    has_w, has_h = "width" in inp, "height" in inp
    if has_w != has_h:
        raise ValidationError(
            "width and height must be sent together.",
            hint="Send both, or neither to use the defaults.",
        )
    width = inp.get("width", config.DEFAULT_WIDTH)
    height = inp.get("height", config.DEFAULT_HEIGHT)
    if not isinstance(width, int) or not isinstance(height, int) \
            or isinstance(width, bool) or isinstance(height, bool):
        raise ValidationError("width and height must be integers.")
    if width < config.MIN_DIMENSION or height < config.MIN_DIMENSION:
        raise ValidationError(
            f"width and height must be at least {config.MIN_DIMENSION}.",
            hint=f"You sent {width}x{height}.",
        )
    if width % config.MOD_VALUE != 0 or height % config.MOD_VALUE != 0:
        snapped_w = width // config.MOD_VALUE * config.MOD_VALUE
        snapped_h = height // config.MOD_VALUE * config.MOD_VALUE
        raise ValidationError(
            f"width and height must be divisible by {config.MOD_VALUE}.",
            hint=f"You sent {width}x{height}. Try {snapped_w}x{snapped_h}.",
        )
    # (config.MOD_VALUE is 16, not 32, so the templates' 1280x720
    # default passes — see config.py.)
    if width * height > config.MAX_RESOLUTION_AREA:
        raise ValidationError(
            f"Total pixels ({width}x{height}={width * height}) exceeds max ({config.MAX_RESOLUTION_AREA}).",
            hint="Use a smaller size or increase MAX_RESOLUTION_AREA.",
        )

    lora_strength = _strength(inp, "lora_strength", config.EXTRA_LORA_STRENGTH)
    lora_strength_2 = _strength(inp, "lora_strength_2", config.EXTRA_LORA_STRENGTH_2)
    lora_strength_3 = _strength(inp, "lora_strength_3", config.EXTRA_LORA_STRENGTH_3)

    output_cfg = inp.get("output", {})
    if not isinstance(output_cfg, dict):
        raise ValidationError("output must be a JSON object.")

    ignored = [p for p in config.IGNORED_PARAMS if p in inp]

    if start_image and last_image:
        mode = "flf"
    elif start_image:
        mode = "i2v"
    elif last_image or ref_images:
        mode = "guides"
    else:
        mode = "t2v"

    return {
        "mode": mode,
        "start_image": start_image,
        "last_image": last_image,
        "ref_images": ref_images,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seconds": seconds,
        "fps": fps,
        "frames": frames,
        "seed": seed,
        "mute": mute,
        "width": width,
        "height": height,
        "lora_strength": lora_strength,
        "lora_strength_2": lora_strength_2,
        "lora_strength_3": lora_strength_3,
        "ignored_params": ignored,
        "output": {
            "bucket": output_cfg.get("bucket", config.S3_BUCKET_DEFAULT),
            "prefix": output_cfg.get("prefix", "ltx-2.5/"),
            "public": output_cfg.get("public", True),
            "ttl_seconds": output_cfg.get("ttl_seconds", 0),
        },
    }
