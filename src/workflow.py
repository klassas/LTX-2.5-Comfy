"""ComfyUI node graph for LTX-2.5, built in code per job.

A faithful transcription of ComfyUI's official LTX-2.5 templates
(Comfy-Org/workflow_templates: video_ltx2_5_t2v / _i2v / _flf2v), minus
the prompt-enhancer branch (a second Gemma 4 encoder behind a
TextGenerateLTX2Prompt node — prompts pass through as typed, same as
the sibling workers). The three templates share one two-stage skeleton;
the I2V/FLF variants differ only in how the video latent is conditioned
before stage 1, so all modes are unified here into the two-stage graph:

    UNETLoader (+ optional LoRA chain) ──> LTXVDualCFGGuider.model (both stages)
    CLIPLoader (type ltxv) ──> CLIPTextEncode x2 ──> LTXVConditioning
    EmptyLTXVLatentVideo (w/2, h/2, frames) ──[start frame / guides]──┐
    LTXVEmptyLatentAudio ──> LTXVConcatAVLatent ──> SamplerCustomAdvanced #1
      ──> LTXVSeparateAVLatent ──[LTXVCropGuides]──> LTXVLatentUpsampler
      ──[start frame again]──> LTXVConcatAVLatent (with stage-1 audio)
      ──> SamplerCustomAdvanced #2 ──> LTXVSeparateAVLatent
      ──> VAEDecodeTiled + LTXVAudioVAEDecode ──> CreateVideo ──> SaveVideo

Conditioning choices, per template:

* Start frame (I2V template): LoadImage -> LTXVPreprocess ->
  LTXVImgToVideoInplace applied in BOTH stages — strength 0.7 in
  stage 1, 1.0 in stage 2 (the template's widget values). The node
  rescales the image to the latent's pixel size itself, so no resize
  node is needed.
* Guides (FLF template): LTXVAddGuide chains on the *video* latent
  (guides cannot be added to a combined AV latent) and threads the
  conditioning through each guide; the stage-1 guider therefore reads
  the LAST guide's positive/negative, exactly as FLF does. The FLF
  template is single-stage and crops the appended guide frames right
  before decoding; here the crop happens after stage 1, before the
  latent upscaler (a guide frame must not be upscaled into stage 2),
  and stage 2's guider reads LTXVCropGuides' cleaned conditioning
  outputs (keyframe entries removed) instead of the guide-threaded
  ones. Also following FLF, stage 1's sampler output is taken from
  `denoised_output` when guides are present (`output` otherwise, as in
  the T2V/I2V templates).
* Mid guides from `ref_images` are spread evenly across the clip:
  guide i of n sits at frame round((i+1)/(n+1) * (frames-1)), snapped
  down to a multiple of 8 (LTXVAddGuide's own frame grid). The last
  frame is a guide at frame_idx=-1, chained last, as in FLF.
* Sampler seeds: stage 1 uses the job seed; stage 2's RandomNoise is
  fixed at 42 in all three templates, so it is fixed here too.

Node schemas verified against ComfyUI v0.46.2 sources
(comfy_extras/nodes_lt.py, nodes_lt_audio.py, nodes_lt_upsampler.py,
nodes_video.py, nodes.py, comfy_extras/nodes_custom_sampler.py,
comfy_extras/nodes_hunyuan.py for LatentUpscaleModelLoader).
"""

# Model files, as laid out in Lightricks/LTX-2.5 (the Model Cache
# snapshot is symlinked into these exact category dirs by
# model_link.py). This is the official templates' distilled INT8 set.
UNET_NAME = "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
CLIP_NAME = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
VAE_NAME = "ltx-2.5-video-vae-bf16.safetensors"
AUDIO_VAE_NAME = "ltx-2.5-audio-vae-bf16.safetensors"
UPSCALER_NAME = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"

OUTPUT_NODE_ID = "save_video"

# Distilled sampler schedules, extracted from the official templates.
STAGE1_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
STAGE2_SIGMAS = "0.85, 0.7250, 0.4219, 0.0"
SAMPLER_NAME = "euler_ancestral"
STAGE2_SEED = 42  # fixed in the official templates

# LTXVPreprocess compression the templates apply to every input image.
IMG_COMPRESSION = 18
# LTXVImgToVideoInplace strengths (I2V template widgets: stage 1, stage 2).
START_STRENGTH_STAGE1 = 0.7
START_STRENGTH_STAGE2 = 1.0
# LTXVAddGuide strength (FLF template).
GUIDE_STRENGTH = 0.7


def _model_branch(nodes: dict, extra_loras: list | None) -> list:
    """Adds the caller's extra LoRAs (up to three, as (name, strength)
    pairs; a None name skips the slot) after the UNETLoader and returns
    the MODEL ref both stage guiders read from."""
    model_ref = ["unet_loader", 0]
    for i, (name, strength) in enumerate(extra_loras or []):
        if not name:
            continue
        node_id = "lora_extra" if i == 0 else f"lora_extra_{i + 1}"
        nodes[node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": model_ref,
                "lora_name": name,
                "strength_model": strength,
            },
        }
        model_ref = [node_id, 0]
    return model_ref


def _guide_frame_idx(index: int, count: int, frames: int) -> int:
    """Even spread of `count` mid guides over `frames` frames: guide
    `index` (0-based) sits at fraction (index+1)/(count+1) of the clip,
    snapped down to LTXVAddGuide's multiple-of-8 frame grid."""
    raw = round((index + 1) / (count + 1) * (frames - 1))
    return max(0, (raw // 8) * 8)


def build_workflow(
    *,
    prompt: str,
    negative_prompt: str = "",
    width: int = 1280,
    height: int = 720,
    frames: int = 121,
    fps: int = 24,
    seed: int = 0,
    start_image_name: str | None = None,
    last_image_name: str | None = None,
    ref_image_names: list[str] | None = None,
    extra_loras: list | None = None,
    mute: bool = False,
    filename_prefix: str = "ltx_25/job",
) -> dict:
    """Returns the API-format graph: {node_id: {"class_type", "inputs"}}.

    start_image_name / last_image_name / ref_image_names: files already
    uploaded to ComfyUI's input/ directory. `frames` must already be on
    the 1 + 8k grid (the validator snaps it). width/height are the
    OUTPUT size; stage 1 runs at half size, per the templates.
    """
    refs = list(ref_image_names or [])
    guides: list[tuple[str, int]] = [
        (name, _guide_frame_idx(i, len(refs), frames)) for i, name in enumerate(refs)
    ]
    if last_image_name:
        guides.append((last_image_name, -1))
    has_guides = bool(guides)

    nodes: dict[str, dict] = {
        "unet_loader": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": UNET_NAME, "weight_dtype": "default"},
        },
        "clip_loader": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": CLIP_NAME, "type": "ltxv", "device": "default"},
        },
        "vae_loader": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": VAE_NAME},
        },
        # The official templates load the audio VAE with a plain
        # VAELoader too (the file lives in vae/). LTXVAudioVAELoader
        # would expect it under checkpoints/ instead — not used.
        "audio_vae_loader": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": AUDIO_VAE_NAME},
        },
        "upscale_model_loader": {
            "class_type": "LatentUpscaleModelLoader",
            "inputs": {"model_name": UPSCALER_NAME},
        },
        "encode_pos": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["clip_loader", 0], "text": prompt},
        },
        "encode_neg": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["clip_loader", 0], "text": negative_prompt},
        },
        "conditioning": {
            "class_type": "LTXVConditioning",
            "inputs": {
                "positive": ["encode_pos", 0],
                "negative": ["encode_neg", 0],
                "frame_rate": float(fps),
            },
        },
        "empty_video": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {
                "width": width // 2,
                "height": height // 2,
                "length": frames,
                "batch_size": 1,
            },
        },
        "empty_audio": {
            "class_type": "LTXVEmptyLatentAudio",
            "inputs": {
                "frames_number": frames,
                "frame_rate": float(fps),
                "batch_size": 1,
                "audio_vae": ["audio_vae_loader", 0],
            },
        },
    }
    model_ref = _model_branch(nodes, extra_loras)

    # -- Video-latent conditioning (start frame, then guides) ──────────
    video_ref: list = ["empty_video", 0]
    pos_ref: list = ["conditioning", 0]
    neg_ref: list = ["conditioning", 1]

    if start_image_name:
        nodes["load_start"] = {
            "class_type": "LoadImage",
            "inputs": {"image": start_image_name},
        }
        nodes["preprocess_start"] = {
            "class_type": "LTXVPreprocess",
            "inputs": {"image": ["load_start", 0], "img_compression": IMG_COMPRESSION},
        }
        nodes["inplace_1"] = {
            "class_type": "LTXVImgToVideoInplace",
            "inputs": {
                "vae": ["vae_loader", 0],
                "image": ["preprocess_start", 0],
                "latent": video_ref,
                "strength": START_STRENGTH_STAGE1,
                "bypass": False,
            },
        }
        video_ref = ["inplace_1", 0]

    for i, (name, frame_idx) in enumerate(guides):
        tag = f"guide_{i + 1}"
        nodes[f"load_{tag}"] = {
            "class_type": "LoadImage",
            "inputs": {"image": name},
        }
        nodes[f"preprocess_{tag}"] = {
            "class_type": "LTXVPreprocess",
            "inputs": {"image": [f"load_{tag}", 0], "img_compression": IMG_COMPRESSION},
        }
        nodes[f"add_{tag}"] = {
            "class_type": "LTXVAddGuide",
            "inputs": {
                "positive": pos_ref,
                "negative": neg_ref,
                "vae": ["vae_loader", 0],
                "latent": video_ref,
                "image": [f"preprocess_{tag}", 0],
                "frame_idx": frame_idx,
                "strength": GUIDE_STRENGTH,
            },
        }
        pos_ref = [f"add_{tag}", 0]
        neg_ref = [f"add_{tag}", 1]
        video_ref = [f"add_{tag}", 2]

    # -- Stage 1 (half resolution) ─────────────────────────────────────
    nodes["concat_1"] = {
        "class_type": "LTXVConcatAVLatent",
        "inputs": {"video_latent": video_ref, "audio_latent": ["empty_audio", 0]},
    }
    nodes["noise_1"] = {
        "class_type": "RandomNoise",
        "inputs": {"noise_seed": seed},
    }
    nodes["sampler_select_1"] = {
        "class_type": "KSamplerSelect",
        "inputs": {"sampler_name": SAMPLER_NAME},
    }
    nodes["sigmas_1"] = {
        "class_type": "ManualSigmas",
        "inputs": {"sigmas": STAGE1_SIGMAS},
    }
    nodes["guider_1"] = {
        "class_type": "LTXVDualCFGGuider",
        "inputs": {
            "model": model_ref,
            "positive": pos_ref,
            "negative": neg_ref,
            "video_cfg": 1.0,
            "audio_cfg": 1.0,
        },
    }
    nodes["sampler_1"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["noise_1", 0],
            "guider": ["guider_1", 0],
            "sampler": ["sampler_select_1", 0],
            "sigmas": ["sigmas_1", 0],
            "latent_image": ["concat_1", 0],
        },
    }
    # FLF reads the sampler's denoised_output when guides are in play;
    # T2V/I2V read output. Slot 1 = denoised_output, slot 0 = output.
    sampler_1_slot = 1 if has_guides else 0
    nodes["separate_1"] = {
        "class_type": "LTXVSeparateAVLatent",
        "inputs": {"av_latent": ["sampler_1", sampler_1_slot]},
    }

    # -- Between stages: crop guides, upscale, re-apply start frame ───
    if has_guides:
        nodes["crop_guides"] = {
            "class_type": "LTXVCropGuides",
            "inputs": {
                "positive": pos_ref,
                "negative": neg_ref,
                "latent": ["separate_1", 0],
            },
        }
        upscale_in: list = ["crop_guides", 2]
        pos2_ref: list = ["crop_guides", 0]
        neg2_ref: list = ["crop_guides", 1]
    else:
        upscale_in = ["separate_1", 0]
        pos2_ref = pos_ref
        neg2_ref = neg_ref

    nodes["upscaler"] = {
        "class_type": "LTXVLatentUpsampler",
        "inputs": {
            "samples": upscale_in,
            "upscale_model": ["upscale_model_loader", 0],
            "vae": ["vae_loader", 0],
        },
    }
    video2_ref: list = ["upscaler", 0]
    if start_image_name:
        nodes["inplace_2"] = {
            "class_type": "LTXVImgToVideoInplace",
            "inputs": {
                "vae": ["vae_loader", 0],
                "image": ["preprocess_start", 0],
                "latent": video2_ref,
                "strength": START_STRENGTH_STAGE2,
                "bypass": False,
            },
        }
        video2_ref = ["inplace_2", 0]

    # -- Stage 2 (full resolution) ─────────────────────────────────────
    nodes["concat_2"] = {
        "class_type": "LTXVConcatAVLatent",
        "inputs": {"video_latent": video2_ref, "audio_latent": ["separate_1", 1]},
    }
    nodes["noise_2"] = {
        "class_type": "RandomNoise",
        "inputs": {"noise_seed": STAGE2_SEED},
    }
    nodes["sampler_select_2"] = {
        "class_type": "KSamplerSelect",
        "inputs": {"sampler_name": SAMPLER_NAME},
    }
    nodes["sigmas_2"] = {
        "class_type": "ManualSigmas",
        "inputs": {"sigmas": STAGE2_SIGMAS},
    }
    nodes["guider_2"] = {
        "class_type": "LTXVDualCFGGuider",
        "inputs": {
            "model": model_ref,
            "positive": pos2_ref,
            "negative": neg2_ref,
            "video_cfg": 1.0,
            "audio_cfg": 1.0,
        },
    }
    nodes["sampler_2"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["noise_2", 0],
            "guider": ["guider_2", 0],
            "sampler": ["sampler_select_2", 0],
            "sigmas": ["sigmas_2", 0],
            "latent_image": ["concat_2", 0],
        },
    }
    nodes["separate_2"] = {
        "class_type": "LTXVSeparateAVLatent",
        "inputs": {"av_latent": ["sampler_2", 0]},
    }

    # -- Decode + mux ──────────────────────────────────────────────────
    nodes["decode_video"] = {
        "class_type": "VAEDecodeTiled",
        "inputs": {
            "samples": ["separate_2", 0],
            "vae": ["vae_loader", 0],
            "tile_size": 512,
            "overlap": 64,
            "temporal_size": 64,
            "temporal_overlap": 16,
        },
    }
    create_inputs: dict = {
        "images": ["decode_video", 0],
        "fps": float(fps),
        "bit_depth": 8,
    }
    if not mute:
        nodes["decode_audio"] = {
            "class_type": "LTXVAudioVAEDecode",
            "inputs": {
                "samples": ["separate_2", 1],
                "audio_vae": ["audio_vae_loader", 0],
            },
        }
        create_inputs["audio"] = ["decode_audio", 0]
    nodes["create_video"] = {
        "class_type": "CreateVideo",
        "inputs": create_inputs,
    }
    nodes[OUTPUT_NODE_ID] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["create_video", 0],
            "filename_prefix": filename_prefix,
            # SaveVideo's format is a DynamicCombo: in API format the
            # selected key goes in `format` and the nested codec
            # selection in `format.codec` ("auto" has no further
            # nested inputs).
            "format": "auto",
            "format.codec": "auto",
        },
    }
    return nodes
