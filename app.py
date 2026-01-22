import os

# Set PyTorch expandable segments BEFORE importing torch (VERY IMPORTANT)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import copy
import time
import random
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Union
import torch
from PIL import Image
import gradio as gr
import spaces

from diffusers import (
    DiffusionPipeline,
    AutoencoderKL,
    AutoPipelineForImage2Image,
    FluxPipeline,
    FlowMatchEulerDiscreteScheduler)

from huggingface_hub import (
    hf_hub_download,
    HfFileSystem,
    ModelCard,
    snapshot_download)
from diffusers.utils import load_image
from typing import Iterable
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

colors.steel_blue = colors.Color(
    name="steel_blue",
    c50="#EBF3F8",
    c100="#D3E5F0",
    c200="#A8CCE1",
    c300="#7DB3D2",
    c400="#529AC3",
    c500="#4682B4",
    c600="#3E72A0",
    c700="#36638C",
    c800="#2E5378",
    c900="#264364",
    c950="#1E3450",
)

class SteelBlueTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.steel_blue,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_800)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_500)",
            button_secondary_text_color="black",
            button_secondary_text_color_hover="white",
            button_secondary_background_fill="linear-gradient(90deg, *primary_300, *primary_300)",
            button_secondary_background_fill_hover="linear-gradient(90deg, *primary_400, *primary_400)",
            button_secondary_background_fill_dark="linear-gradient(90deg, *primary_500, *primary_600)",
            button_secondary_background_fill_hover_dark="linear-gradient(90deg, *primary_500, *primary_500)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

steel_blue_theme = SteelBlueTheme()

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

@torch.inference_mode()
def flux_pipe_call_that_returns_an_iterable_of_images(
    self,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    timesteps: List[int] = None,
    guidance_scale: float = 3.5,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    max_sequence_length: int = 512,
    good_vae: Optional[Any] = None,
):
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor
    
    self.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    batch_size = 1 if isinstance(prompt, str) else len(prompt)
    device = self._execution_device

    lora_scale = joint_attention_kwargs.get("scale", None) if joint_attention_kwargs is not None else None
    prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )
    
    num_channels_latents = self.transformer.config.in_channels // 4
    latents, latent_image_ids = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
    
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.base_image_seq_len,
        self.scheduler.config.max_image_seq_len,
        self.scheduler.config.base_shift,
        self.scheduler.config.max_shift,
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        timesteps,
        sigmas,
        mu=mu,
    )
    self._num_timesteps = len(timesteps)

    guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32).expand(latents.shape[0]) if self.transformer.config.guidance_embeds else None

    for i, t in enumerate(timesteps):
        if self.interrupt:
            continue

        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        noise_pred = self.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=self.joint_attention_kwargs,
            return_dict=False,
        )[0]

        # REMOVE decoding inside the loop completely (MANDATORY FIX)
        # Decoding every step is fatal on RTX 4070 - causes massive memory allocation
        # Only keep the scheduler step
        # ❌ Remove torch.cuda.empty_cache() inside loop - hurts performance and increases fragmentation
        latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
    # Decode only once at the end (after all steps complete)
    latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
    latents = (latents / good_vae.config.scaling_factor) + good_vae.config.shift_factor
    image = good_vae.decode(latents, return_dict=False)[0]
    self.maybe_free_model_hooks()
    # ✅ FIX: Removed torch.cuda.empty_cache() here - with CPU offload, this forces synchronization
    # and can increase peak VRAM during offload transitions. Memory is cleaned outside safely.
    yield self.image_processor.postprocess(image, output_type=output_type)[0]

# ✅ OFFLINE MODE: Setup local LoRA cache directory (define early for use below)
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load LoRAs from external JSON file
with open("loras.json", "r", encoding="utf-8") as f:
    loras = json.load(f)

# ✅ OFFLINE MODE: Convert local image paths to absolute paths for Gradio
# Gradio needs absolute paths for local images, but can handle URLs directly
def get_image_path_or_url(image_path_or_url):
    """Convert local image paths to absolute paths, keep URLs as-is."""
    if image_path_or_url.startswith("http://") or image_path_or_url.startswith("https://"):
        # It's a URL, return as-is
        return image_path_or_url
    else:
        # It's a local path, convert to absolute
        if not os.path.isabs(image_path_or_url):
            # Relative path - make it absolute
            return os.path.join(WORKSPACE_DIR, image_path_or_url)
        return image_path_or_url

# Update all image paths in loras to use absolute paths
for lora in loras:
    if "image" in lora and lora["image"]:
        lora["image"] = get_image_path_or_url(lora["image"])
# ✅ OFFLINE MODE: Setup local cache directories
MODELS_CACHE_DIR = os.path.join(WORKSPACE_DIR, "models_cache")
LORAS_CACHE_DIR = os.path.join(WORKSPACE_DIR, "loras_cache")
os.makedirs(MODELS_CACHE_DIR, exist_ok=True)
os.makedirs(LORAS_CACHE_DIR, exist_ok=True)
print(f"✓ Models cache directory: {MODELS_CACHE_DIR}")
print(f"✓ LoRA cache directory: {LORAS_CACHE_DIR}")

def get_lora_local_path(lora_repo_or_path):
    """
    Get the local cache path for a LoRA.
    Handles both HuggingFace repo IDs and local cache paths.
    
    Args:
        lora_repo_or_path: Either a HuggingFace repo ID (e.g., "user/repo") 
                          or a local cache path (e.g., "loras_cache/user_repo")
    
    Returns:
        Absolute path to cached LoRA directory
    """
    # Check if it's already a local path (starts with loras_cache/)
    if lora_repo_or_path.startswith("loras_cache/") or os.path.isabs(lora_repo_or_path):
        # It's already a local path
        if os.path.isabs(lora_repo_or_path):
            return lora_repo_or_path
        else:
            # Relative path - make it absolute
            return os.path.join(WORKSPACE_DIR, lora_repo_or_path)
    else:
        # It's a HuggingFace repo ID - convert to local cache path
        safe_name = lora_repo_or_path.replace("/", "_").replace(".", "_")
        return os.path.join(LORAS_CACHE_DIR, safe_name)

def ensure_lora_cached(lora_repo_or_path, weight_name=None, offline_mode=False):
    """
    Ensure LoRA is cached locally. Downloads if not present (unless offline_mode=True).
    Returns the local path to the cached LoRA.
    
    Args:
        lora_repo_or_path: Either a HuggingFace repo ID (e.g., "user/repo") 
                          or a local cache path (e.g., "loras_cache/user_repo")
        weight_name: Optional specific weight file name
        offline_mode: If True, only check cache, don't download
    
    Returns:
        Local path to cached LoRA directory
    """
    local_path = get_lora_local_path(lora_repo_or_path)
    
    # If it's already a local path and exists, return it directly (offline mode)
    if (lora_repo_or_path.startswith("loras_cache/") or os.path.isabs(lora_repo_or_path)):
        if os.path.exists(local_path) and os.path.isdir(local_path):
            # Check if it has the required files
            if weight_name:
                weight_path = os.path.join(local_path, weight_name)
                if os.path.exists(weight_path):
                    return local_path
            else:
                # Check for any safetensors file
                try:
                    files = os.listdir(local_path)
                    if any(f.endswith(".safetensors") for f in files):
                        return local_path
                except:
                    pass
        # Local path doesn't exist - this shouldn't happen if download script worked
        if offline_mode:
            raise FileNotFoundError(f"LoRA cache path not found: {local_path}")
        # Fall through to try downloading (in case path format changed)
    
    # Original repo ID path - check cache or download
    
    # Check if already cached
    if os.path.exists(local_path) and os.path.isdir(local_path):
        # Check if it has the required files
        if weight_name:
            weight_path = os.path.join(local_path, weight_name)
            if os.path.exists(weight_path):
                print(f"✓ LoRA already cached: {lora_repo}")
                return local_path
        else:
            # Check for any safetensors file
            files = os.listdir(local_path)
            if any(f.endswith(".safetensors") for f in files):
                print(f"✓ LoRA already cached: {lora_repo}")
                return local_path
    
    # Not cached - download if not in offline mode
    if offline_mode:
        raise FileNotFoundError(
            f"LoRA '{lora_repo}' not found in cache and offline mode is enabled. "
            f"Please download it first or disable offline mode."
        )
    
    print(f"Downloading LoRA to cache: {lora_repo}")
    try:
        # Download entire LoRA repository to cache
        cached_path = snapshot_download(
            repo_id=lora_repo,
            cache_dir=LORAS_CACHE_DIR,
            local_dir=local_path,
            local_dir_use_symlinks=False,  # Copy files, don't symlink
            resume_download=True
        )
        print(f"✓ LoRA cached successfully: {lora_repo}")
        return local_path
    except Exception as e:
        print(f"❌ Failed to download LoRA '{lora_repo}': {e}")
        raise

# Minimal "safe" configuration for RTX 4070 (12GB VRAM)
# Use FP16, not bfloat16 on RTX 4070
# RTX 4070 does not benefit from bfloat16 and often uses more memory
# This alone saves ~10–15% memory
dtype = torch.float16
device = "cuda" if torch.cuda.is_available() else "cpu"
base_model = "black-forest-labs/FLUX.1-dev"

# Use single VAE (AutoencoderKL) for both pipelines
# ✅ FIX: Only load ONE VAE (not both Tiny + KL) - loading two VAEs is too much for RTX 4070
# taef1 (AutoencoderTiny) has been removed - use only good_vae (AutoencoderKL)
# ❌ DO NOT use .to(device) here - CPU offload manages device placement automatically
# ✅ FIX: Lazy loading for pipe_i2i to prevent OOM at startup
# Loading both pipelines at once is too much for 12GB RTX 4070
# Load pipe_i2i only when needed (first image-to-image request)
# ✅ OFFLINE MODE: Use local models_cache directory instead of default C drive cache
good_vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=dtype, low_cpu_mem_usage=True, cache_dir=MODELS_CACHE_DIR)
print("Loading text-to-image pipeline...")
try:
    pipe = DiffusionPipeline.from_pretrained(base_model, torch_dtype=dtype, vae=good_vae, low_cpu_mem_usage=True, cache_dir=MODELS_CACHE_DIR)
    print("✓ Text-to-image pipeline loaded")
except Exception as e:
    print(f"❌ Failed to load text-to-image pipeline: {e}")
    raise

# Enable memory-efficient attention
# xformers: Memory-efficient attention implementation
try:
    pipe.enable_xformers_memory_efficient_attention()
    print("✓ xformers enabled")
except Exception as e:
    print(f"⚠ xformers not available: {e}")

# ✅ FIX: FLUX does not benefit from attention slicing the way SDXL does
# This line actually increases peak VRAM on FLUX
# We already have: xformers + CPU offload + FP16 - that's enough
# pipe.enable_attention_slicing()  # ❌ Removed - increases VRAM on FLUX

# Sequential CPU offload: Moves models to CPU sequentially when not in use
# ⚠️ Sequential offload is slower than pure GPU but does not cause the same peak spikes
# during transformer + scheduler stepping loops (critical for FLUX + custom loop on 12GB)
pipe.enable_sequential_cpu_offload()
print("✓ Sequential CPU offload enabled (prevents peak VRAM spikes)")

# ✅ LAZY LOADING: pipe_i2i will be loaded on-demand when first image-to-image is requested
pipe_i2i = None

MAX_SEED = 2**32-1
pipe.flux_pipe_call_that_returns_an_iterable_of_images = flux_pipe_call_that_returns_an_iterable_of_images.__get__(pipe)

def ensure_pipe_i2i_loaded():
    """Lazy load pipe_i2i only when needed to prevent OOM at startup."""
    global pipe_i2i, _current_lora_path
    if pipe_i2i is None:
        print("Loading image-to-image pipeline (lazy load)...")
        try:
            # Clear cache before loading second pipeline
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
            # ✅ FIX: Do NOT share transformer between pipelines - causes instability with LoRA + adapter registry + offload
            # Load pipe_i2i independently (uses more RAM on CPU, but avoids GPU spikes and adapter conflicts)
            # ✅ OFFLINE MODE: Use local models_cache directory instead of default C drive cache
            pipe_i2i = AutoPipelineForImage2Image.from_pretrained(
                base_model,
                vae=good_vae,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                cache_dir=MODELS_CACHE_DIR
            )
            
            # Enable optimizations
            try:
                pipe_i2i.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
            # pipe_i2i.enable_attention_slicing()  # ❌ Removed - increases VRAM on FLUX
            pipe_i2i.enable_sequential_cpu_offload()
            
            print("✓ Image-to-image pipeline loaded")
            
            # ✅ If a LoRA is already loaded on pipe, load it on pipe_i2i too
            if _current_lora_path is not None:
                print(f"Loading current LoRA on image-to-image pipeline: {_current_lora_path}")
                try:
                    # Find the LoRA entry
                    selected_lora = None
                    for lora in loras:
                        if lora["repo"] == _current_lora_path:
                            selected_lora = lora
                            break
                    
                    if selected_lora:
                        weight_name = selected_lora.get("weights", None)
                        # ✅ OFFLINE MODE: Get cached local path for LoRA
                        try:
                            cached_lora_path = ensure_lora_cached(_current_lora_path, weight_name, offline_mode=True)
                            lora_path_to_load = cached_lora_path
                        except FileNotFoundError:
                            # Not in cache - download it (or use online if network available)
                            cached_lora_path = ensure_lora_cached(_current_lora_path, weight_name, offline_mode=False)
                            lora_path_to_load = cached_lora_path
                        
                        # ✅ FIX: Sanitize adapter name - PyTorch module names cannot contain "." or "/"
                        # Replace both "/" and "." with "_" to create valid module names
                        adapter_name = f"lora_{selected_lora['repo'].replace('/', '_').replace('.', '_')}"
                        pipe_i2i.load_lora_weights(
                            lora_path_to_load,
                            weight_name=weight_name,
                            adapter_name=adapter_name,
                            prefix=None,
                            low_cpu_mem_usage=True
                        )
                        print("✓ LoRA loaded on image-to-image pipeline")
                except Exception as e:
                    print(f"⚠ Failed to load LoRA on image-to-image pipeline: {e}")
                    
        except Exception as e:
            print(f"❌ Failed to load image-to-image pipeline: {e}")
            raise

# Track currently loaded LoRA to avoid unnecessary unload/load cycles
_current_lora_path = None
# Track LoRA switches and generations for pipeline restart to prevent memory fragmentation
_lora_switch_count = 0
_generation_count = 0
# Thresholds for pipeline restart (prevents long-session memory fragmentation)
LORA_SWITCH_THRESHOLD = 3  # Restart after 3+ LoRA switches
GENERATION_THRESHOLD = 12  # Restart every 12 generations

def nuke_all_adapters(p):
    """
    Aggressively removes ALL LoRA adapters from the Diffusers pipeline registry.
    This prevents adapter garbage from accumulating and causing hidden GPU allocations.
    Diffusers keeps a registry of all adapters, not just "active" ones.
    """
    try:
        if hasattr(p, "get_list_adapters"):
            for name in p.get_list_adapters():
                try:
                    p.delete_adapter(name)
                except Exception:
                    pass
    except Exception:
        pass

def restart_pipelines():
    """Restart pipelines to prevent memory fragmentation from repeated LoRA switches."""
    global pipe, pipe_i2i, good_vae, _current_lora_path, _lora_switch_count, _generation_count
    
    print("🔄 Restarting pipelines to prevent memory fragmentation...")
    with calculateDuration("Restarting pipelines"):
        # Delete old pipelines
        del pipe
        if pipe_i2i is not None:
            del pipe_i2i
            pipe_i2i = None
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        # Reload text-to-image pipeline
        # ✅ OFFLINE MODE: Use local models_cache directory instead of default C drive cache
        good_vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=dtype, low_cpu_mem_usage=True, cache_dir=MODELS_CACHE_DIR)
        pipe = DiffusionPipeline.from_pretrained(base_model, torch_dtype=dtype, vae=good_vae, low_cpu_mem_usage=True, cache_dir=MODELS_CACHE_DIR)
        
        # Re-enable optimizations
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        # ✅ FIX: FLUX does not benefit from attention slicing - removed to prevent VRAM increase
        # pipe.enable_attention_slicing()  # ❌ Removed
        pipe.enable_sequential_cpu_offload()
        
        # Re-attach custom function
        pipe.flux_pipe_call_that_returns_an_iterable_of_images = flux_pipe_call_that_returns_an_iterable_of_images.__get__(pipe)
        
        # Reset tracking variables
        _current_lora_path = None
        _lora_switch_count = 0
        _generation_count = 0
        
        torch.cuda.empty_cache()
        print("✓ Pipelines restarted successfully (pipe_i2i will be loaded on-demand)")

class calculateDuration:
    def __init__(self, activity_name=""):
        self.activity_name = activity_name

    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        if self.activity_name:
            print(f"Elapsed time for {self.activity_name}: {self.elapsed_time:.6f} seconds")
        else:
            print(f"Elapsed time: {self.elapsed_time:.6f} seconds")

def update_selection(evt: gr.SelectData, width, height):
    selected_lora = loras[evt.index]
    new_placeholder = f"Type a prompt for {selected_lora['title']}"
    lora_repo = selected_lora["repo"]
    updated_text = f"### Selected: [{lora_repo}](https://huggingface.co/{lora_repo}) ✅"
    if "aspect" in selected_lora:
        if selected_lora["aspect"] == "portrait":
            width = 768
            height = 1024  # 768x1024 is safer than 1024x1024
        elif selected_lora["aspect"] == "landscape":
            width = 1024
            height = 768  # 1024x768 is safer than 1024x1024
        else:
            width = 768
            height = 768  # 768x768 is stable for 12GB VRAM
    return (
        gr.update(placeholder=new_placeholder),
        updated_text,
        evt.index,
        width,
        height,
    )

@spaces.GPU
def generate_image(prompt_mash, steps, seed, cfg_scale, width, height, lora_scale, progress):
    # Check if pipeline restart is needed (prevents long-session memory fragmentation)
    global _lora_switch_count, _generation_count
    if _lora_switch_count >= LORA_SWITCH_THRESHOLD or _generation_count >= GENERATION_THRESHOLD:
        restart_pipelines()
    
    # Safety check: Abort if GPU memory is too high to prevent PC freeze
    if torch.cuda.memory_allocated() > 10 * 1024**3:
        raise RuntimeError("GPU memory too high — aborting to avoid crash")
    
    # CPU offload handles device placement automatically
    # ✅ FIX: Use CPU generator when using enable_sequential_cpu_offload() - prevents VRAM climb
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with calculateDuration("Generating image"):
        # Generate image
        for img in pipe.flux_pipe_call_that_returns_an_iterable_of_images(
            prompt=prompt_mash,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            width=width,
            height=height,
            generator=generator,
            joint_attention_kwargs={"scale": lora_scale},
            output_type="pil",
            good_vae=good_vae,
        ):
            yield img
    
    # Increment generation count for pipeline restart tracking
    _generation_count += 1

def generate_image_to_image(prompt_mash, image_input_path, image_strength, steps, cfg_scale, width, height, lora_scale, seed):
    # ✅ LAZY LOAD: Ensure pipe_i2i is loaded before use
    ensure_pipe_i2i_loaded()
    
    # Check if pipeline restart is needed (prevents long-session memory fragmentation)
    global _lora_switch_count, _generation_count
    if _lora_switch_count >= LORA_SWITCH_THRESHOLD or _generation_count >= GENERATION_THRESHOLD:
        restart_pipelines()
        # Re-ensure pipe_i2i after restart
        ensure_pipe_i2i_loaded()
    
    # Safety check: Abort if GPU memory is too high to prevent PC freeze
    if torch.cuda.memory_allocated() > 10 * 1024**3:
        raise RuntimeError("GPU memory too high — aborting to avoid crash")
    
    # ✅ HARD SAFETY CAP: Clamp resolution automatically (12GB safe)
    if width * height > 768 * 1024:
        width, height = 768, 1024
    
    # CPU offload handles device placement automatically
    # ✅ FIX: Use CPU generator when using enable_sequential_cpu_offload() - prevents VRAM climb
    generator = torch.Generator(device="cpu").manual_seed(seed)
    image_input = load_image(image_input_path)
    final_image = pipe_i2i(
        prompt=prompt_mash,
        image=image_input,
        strength=image_strength,
        num_inference_steps=steps,
        guidance_scale=cfg_scale,
        width=width,
        height=height,
        generator=generator,
        joint_attention_kwargs={"scale": lora_scale},
        output_type="pil",
    ).images[0]
    
    # Increment generation count for pipeline restart tracking
    _generation_count += 1
    
    return final_image 

@spaces.GPU
def run_lora(prompt, image_input, image_strength, cfg_scale, steps, selected_index, randomize_seed, seed, width, height, lora_scale, progress=gr.Progress(track_tqdm=True)):
    # Safety check: Abort if GPU memory is too high to prevent PC freeze
    if torch.cuda.memory_allocated() > 10 * 1024**3:
        raise RuntimeError("GPU memory too high — aborting to avoid crash")
    
    # Warning: 1024x1024 + LoRA is risky on 12GB VRAM
    if width == 1024 and height == 1024:
        raise gr.Warning("⚠️ 1024×1024 + LoRA on 12GB VRAM is risky. Consider using 768×768 for stable operation.")
    
    if selected_index is None:
        raise gr.Error("You must select a LoRA before proceeding.")
    selected_lora = loras[selected_index]
    lora_path = selected_lora["repo"]
    trigger_word = selected_lora["trigger_word"]
    if(trigger_word):
        if "trigger_position" in selected_lora:
            if selected_lora["trigger_position"] == "prepend":
                prompt_mash = f"{trigger_word} {prompt}"
            else:
                prompt_mash = f"{prompt} {trigger_word}"
        else:
            prompt_mash = f"{trigger_word} {prompt}"
    else:
        prompt_mash = prompt

    # ✅ HARD SAFETY CAP: Clamp steps when LoRA is active (prevents PC crash)
    # You can still expose 50 in UI, but clamp internally
    steps = min(steps, 28)

    # ✅ FIX: Avoid repeated unload/load to prevent memory fragmentation
    # Only unload if switching to a different LoRA
    global _current_lora_path, _lora_switch_count, _generation_count
    
    # ✅ LAZY LOAD: Ensure pipe_i2i is loaded if image-to-image is requested
    if image_input is not None:
        ensure_pipe_i2i_loaded()
    pipe_to_use = pipe_i2i if image_input is not None else pipe
    
    # Check if pipeline restart is needed (prevents long-session memory fragmentation)
    if _lora_switch_count >= LORA_SWITCH_THRESHOLD or _generation_count >= GENERATION_THRESHOLD:
        restart_pipelines()
    
    if _current_lora_path != lora_path:
        # Different LoRA - need to unload previous and load new
        _lora_switch_count += 1
        with calculateDuration("Unloading previous LoRA"):
            if _current_lora_path is not None:
                pipe.unload_lora_weights()
                if pipe_i2i is not None:
                    pipe_i2i.unload_lora_weights()
                # ✅ MANDATORY: Don't keep old adapters in memory - delete them explicitly
                # This prevents adapter garbage from accumulating and causing hidden GPU allocations
                nuke_all_adapters(pipe)
                if pipe_i2i is not None:
                    nuke_all_adapters(pipe_i2i)
                # Aggressive memory clearing after unload
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        
        # ✅ MANDATORY: Nuke all adapters before loading new one
        # This prevents adapter garbage from accumulating and causing hidden GPU allocations
        nuke_all_adapters(pipe)
        if pipe_i2i is not None:
            nuke_all_adapters(pipe_i2i)

        with calculateDuration(f"Loading LoRA weights for {selected_lora['title']}"):
            weight_name = selected_lora.get("weights", None)
            # ✅ OFFLINE MODE: Get cached local path for LoRA
            # Try offline mode first, fallback to online if not cached
            try:
                cached_lora_path = ensure_lora_cached(lora_path, weight_name, offline_mode=True)
                lora_path_to_load = cached_lora_path
                print(f"Using cached LoRA: {cached_lora_path}")
            except FileNotFoundError:
                # Not in cache - download it (or use online if network available)
                print(f"LoRA not in cache, downloading...")
                cached_lora_path = ensure_lora_cached(lora_path, weight_name, offline_mode=False)
                lora_path_to_load = cached_lora_path
            
            # ✅ MINIMAL PATCH: Use unique adapter name per LoRA to prevent conflicts
            # Each LoRA gets its own adapter name based on repo, preventing "adapter already in use" errors
            # ✅ FIX: Sanitize adapter name - PyTorch module names cannot contain "." or "/"
            # Replace both "/" and "." with "_" to create valid module names
            adapter_name = f"lora_{selected_lora['repo'].replace('/', '_').replace('.', '_')}"
            # Load with low_cpu_mem_usage to reduce fragmentation
            # ✅ FIX: Load on both pipelines since we no longer share the transformer
            # ✅ OFFLINE MODE: Load from local cache path
            pipe.load_lora_weights(
                lora_path_to_load, 
                weight_name=weight_name,
                adapter_name=adapter_name,
                prefix=None,  # Silences CLIP warning for FLUX LoRAs that don't touch text encoders
                low_cpu_mem_usage=True
            )
            # Only load LoRA on pipe_i2i if it's already loaded (lazy loading)
            if pipe_i2i is not None:
                pipe_i2i.load_lora_weights(
                    lora_path_to_load, 
                    weight_name=weight_name,
                    adapter_name=adapter_name,
                    prefix=None,  # Silences CLIP warning for FLUX LoRAs that don't touch text encoders
                    low_cpu_mem_usage=True
                )
            _current_lora_path = lora_path
            # Aggressive memory clearing after load
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    else:
        # Same LoRA already loaded - no need to reload
        print(f"LoRA {lora_path} already loaded, skipping reload")
            
    with calculateDuration("Randomizing seed"):
        if randomize_seed:
            seed = random.randint(0, MAX_SEED)
            
    if(image_input is not None):
        
        final_image = generate_image_to_image(prompt_mash, image_input, image_strength, steps, cfg_scale, width, height, lora_scale, seed)
        yield final_image, seed, gr.update(visible=False)
    else:
        image_generator = generate_image(prompt_mash, steps, seed, cfg_scale, width, height, lora_scale, progress)
    
        final_image = None
        step_counter = 0
        for image in image_generator:
            step_counter+=1
            final_image = image
            progress_bar = f'<div class="progress-container"><div class="progress-bar" style="--current: {step_counter}; --total: {steps};"></div></div>'
            yield image, seed, gr.update(value=progress_bar, visible=True)
            
        yield final_image, seed, gr.update(value=progress_bar, visible=False)
        
def get_huggingface_safetensors(link):
  split_link = link.split("/")
  if(len(split_link) == 2):
            model_card = ModelCard.load(link)
            base_model = model_card.data.get("base_model")
            print(base_model)
            if((base_model != "black-forest-labs/FLUX.1-dev") and (base_model != "black-forest-labs/FLUX.1-schnell")):
                raise Exception("Flux LoRA Not Found!")
            image_path = model_card.data.get("widget", [{}])[0].get("output", {}).get("url", None)
            trigger_word = model_card.data.get("instance_prompt", "")
            image_url = f"https://huggingface.co/{link}/resolve/main/{image_path}" if image_path else None
            fs = HfFileSystem()
            try:
                list_of_files = fs.ls(link, detail=False)
                for file in list_of_files:
                    if(file.endswith(".safetensors")):
                        safetensors_name = file.split("/")[-1]
                    if (not image_url and file.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))):
                      image_elements = file.split("/")
                      image_url = f"https://huggingface.co/{link}/resolve/main/{image_elements[-1]}"
            except Exception as e:
              print(e)
              gr.Warning(f"You didn't include a link neither a valid Hugging Face repository with a *.safetensors LoRA")
              raise Exception(f"You didn't include a link neither a valid Hugging Face repository with a *.safetensors LoRA")
            return split_link[1], link, safetensors_name, trigger_word, image_url

def check_custom_model(link):
    if(link.startswith("https://")):
        if(link.startswith("https://huggingface.co") or link.startswith("https://www.huggingface.co")):
            link_split = link.split("huggingface.co/")
            return get_huggingface_safetensors(link_split[1])
    else: 
        return get_huggingface_safetensors(link)

def add_custom_lora(custom_lora):
    global loras
    if(custom_lora):
        try:
            title, repo, path, trigger_word, image = check_custom_model(custom_lora)
            print(f"Loaded custom LoRA: {repo}")
            card = f'''
            <div class="custom_lora_card">
              <span>Loaded custom LoRA:</span>
              <div class="card_internal">
                <img src="{image}" />
                <div>
                    <h3>{title}</h3>
                    <small>{"Using: <code><b>"+trigger_word+"</code></b> as the trigger word" if trigger_word else "No trigger word found. If there's a trigger word, include it in your prompt"}<br></small>
                </div>
              </div>
            </div>
            '''
            existing_item_index = next((index for (index, item) in enumerate(loras) if item['repo'] == repo), None)
            if(not existing_item_index):
                new_item = {
                    "image": image,
                    "title": title,
                    "repo": repo,
                    "weights": path,
                    "trigger_word": trigger_word
                }
                print(new_item)
                existing_item_index = len(loras)
                loras.append(new_item)
        
            return gr.update(visible=True, value=card), gr.update(visible=True), gr.Gallery(selected_index=None), f"Custom: {path}", existing_item_index, trigger_word
        except Exception as e:
            gr.Warning(f"Invalid LoRA: either you entered an invalid link, or a non-FLUX LoRA")
            return gr.update(visible=True, value=f"Invalid LoRA: either you entered an invalid link, a non-FLUX LoRA"), gr.update(visible=False), gr.update(), "", None, ""
    else:
        return gr.update(visible=False), gr.update(visible=False), gr.update(), "", None, ""

def remove_custom_lora():
    return gr.update(visible=False), gr.update(visible=False), gr.update(), "", None, ""

run_lora.zerogpu = True

css = '''
#gen_btn{height: 100%}
#gen_column{align-self: stretch}
#title{text-align: center}
#title h1{font-size: 3em; display:inline-flex; align-items:center}
#title img{width: 100px; margin-right: 0.5em}
#gallery .grid-wrap{height: 10vh}
#lora_list{background: var(--block-background-fill);padding: 0 1em .3em; font-size: 90%}
.card_internal{display: flex;height: 100px;margin-top: .5em}
.card_internal img{margin-right: 1em}
.styler{--form-gap-width: 0px !important}
#progress{height:30px}
#progress .generating{display:none}
.progress-container {width: 100%;height: 30px;background-color: #f0f0f0;border-radius: 15px;overflow: hidden;margin-bottom: 20px}
.progress-bar {height: 100%;background-color: #4f46e5;width: calc(var(--current) / var(--total) * 100%);transition: width 0.5s ease-in-out}
'''

with gr.Blocks(delete_cache=(60, 60)) as demo:
    title = gr.HTML("""<h1>FLUX LoRA DLC</h1>""", elem_id="title",)
    selected_index = gr.State(None)
    with gr.Row():
        with gr.Column(scale=3):
            prompt = gr.Textbox(label="Enter Prompt", lines=1, placeholder="✦︎ Choose the LoRA and type the prompt")
        with gr.Column(scale=1, elem_id="gen_column"):
            generate_button = gr.Button("Generate", variant="primary", elem_id="gen_btn")
    with gr.Row():
        with gr.Column():
            selected_info = gr.Markdown("")
            gallery = gr.Gallery(
                [(item["image"], item["title"]) for item in loras],
                label="250+ LoRA DLC's",
                allow_preview=False,
                columns=3,
                elem_id="gallery",
                #show_share_button=False
            )
            with gr.Group():
                custom_lora = gr.Textbox(label="Enter Custom LoRA", placeholder="prithivMLmods/Canopus-LoRA-Flux-Anime")
                gr.Markdown("[Check the list of FLUX LoRA's](https://huggingface.co/models?other=base_model:adapter:black-forest-labs/FLUX.1-dev)", elem_id="lora_list")
            custom_lora_info = gr.HTML(visible=False)
            custom_lora_button = gr.Button("Remove custom LoRA", visible=False)
        with gr.Column():
            progress_bar = gr.Markdown(elem_id="progress",visible=False)
            result = gr.Image(label="Generated Image", format="png", height=610)

    with gr.Row():
        with gr.Accordion("Advanced Settings", open=False):
            with gr.Row():
                input_image = gr.Image(label="Input image", type="filepath")
                image_strength = gr.Slider(label="Denoise Strength", info="Lower means more image influence", minimum=0.1, maximum=1.0, step=0.01, value=0.75)
            with gr.Column():
                with gr.Row():
                    cfg_scale = gr.Slider(label="CFG Scale", minimum=1, maximum=20, step=0.5, value=3.5)
                    steps = gr.Slider(label="Steps", minimum=1, maximum=50, step=1, value=24)
                
                with gr.Row():
                    width = gr.Slider(label="Width", minimum=256, maximum=1536, step=64, value=768)
                    height = gr.Slider(label="Height", minimum=256, maximum=1536, step=64, value=768)
                
                with gr.Row():
                    randomize_seed = gr.Checkbox(True, label="Randomize seed")
                    seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0, randomize=True)
                    lora_scale = gr.Slider(label="LoRA Scale", minimum=0, maximum=3, step=0.01, value=0.95)

    gallery.select(
        update_selection,
        inputs=[width, height],
        outputs=[prompt, selected_info, selected_index, width, height]
    )
    custom_lora.input(
        add_custom_lora,
        inputs=[custom_lora],
        outputs=[custom_lora_info, custom_lora_button, gallery, selected_info, selected_index, prompt]
    )
    custom_lora_button.click(
        remove_custom_lora,
        outputs=[custom_lora_info, custom_lora_button, gallery, selected_info, selected_index, custom_lora]
    )
    gr.on(
        triggers=[generate_button.click, prompt.submit],
        fn=run_lora,
        inputs=[prompt, input_image, image_strength, cfg_scale, steps, selected_index, randomize_seed, seed, width, height, lora_scale],
        outputs=[result, seed, progress_bar]
    )

demo.queue()
demo.launch(theme=steel_blue_theme, css=css, mcp_server=True, ssr_mode=False, show_error=True)