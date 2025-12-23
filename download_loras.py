"""
Script to download all LoRA images and models to local cache for offline use.
This script downloads:
1. All LoRA images to loras_images/ folder
2. All LoRA models to loras_cache/ folder
"""

import os
import json
import requests
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import snapshot_download
import time

# Setup directories
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
LORAS_CACHE_DIR = os.path.join(WORKSPACE_DIR, "loras_cache")
LORAS_IMAGES_DIR = os.path.join(WORKSPACE_DIR, "loras_images")
os.makedirs(LORAS_CACHE_DIR, exist_ok=True)
os.makedirs(LORAS_IMAGES_DIR, exist_ok=True)

def get_lora_local_path(lora_repo):
    """Get the local cache path for a LoRA."""
    safe_name = lora_repo.replace("/", "_").replace(".", "_")
    return os.path.join(LORAS_CACHE_DIR, safe_name)

def download_image(image_url, output_path, max_retries=3):
    """Download an image from URL to local path."""
    for attempt in range(max_retries):
        try:
            response = requests.get(image_url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Get file extension from URL or content type
            ext = ".png"  # default
            if "." in image_url:
                ext = "." + image_url.split(".")[-1].split("?")[0].lower()
            if ext not in [".png", ".jpg", ".jpeg", ".webp"]:
                ext = ".png"
            
            # Update output path with correct extension
            if not output_path.endswith(ext):
                output_path = output_path.rsplit(".", 1)[0] + ext
            
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return output_path
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                print(f"  ❌ Failed to download image after {max_retries} attempts: {e}")
                return None
    return None

def download_lora(repo_id, weight_name=None):
    """Download a LoRA model to cache."""
    local_path = get_lora_local_path(repo_id)
    
    # Check if already cached
    if os.path.exists(local_path) and os.path.isdir(local_path):
        if weight_name:
            weight_path = os.path.join(local_path, weight_name)
            if os.path.exists(weight_path):
                return local_path, True  # Already cached
        else:
            # Check for any safetensors file
            try:
                files = os.listdir(local_path)
                if any(f.endswith(".safetensors") for f in files):
                    return local_path, True  # Already cached
            except:
                pass
    
    # Download LoRA
    try:
        snapshot_download(
            repo_id=repo_id,
            cache_dir=LORAS_CACHE_DIR,
            local_dir=local_path,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        return local_path, False  # Just downloaded
    except Exception as e:
        print(f"  ❌ Failed to download LoRA: {e}")
        return None, False

def get_image_filename(image_url, title):
    """Generate a safe filename for the image."""
    # Extract extension from URL
    ext = ".png"
    if "." in image_url:
        url_ext = "." + image_url.split(".")[-1].split("?")[0].lower()
        if url_ext in [".png", ".jpg", ".jpeg", ".webp"]:
            ext = url_ext
    
    # Create safe filename from title
    safe_title = "".join(c for c in title if c.isalnum() or c in (" ", "-", "_")).strip()
    safe_title = safe_title.replace(" ", "_")
    
    # Use repo name as fallback if title is empty
    if not safe_title:
        safe_title = "image"
    
    return f"{safe_title}{ext}"

def main():
    """Main function to download all LoRAs and images."""
    print("=" * 60)
    print("LoRA Downloader - Offline Cache Builder")
    print("=" * 60)
    print(f"LoRA cache: {LORAS_CACHE_DIR}")
    print(f"Images cache: {LORAS_IMAGES_DIR}")
    print()
    
    # Load loras.json
    loras_json_path = os.path.join(WORKSPACE_DIR, "loras.json")
    if not os.path.exists(loras_json_path):
        print(f"❌ Error: {loras_json_path} not found!")
        return
    
    print(f"Loading {loras_json_path}...")
    with open(loras_json_path, "r", encoding="utf-8") as f:
        loras = json.load(f)
    
    total = len(loras)
    print(f"Found {total} LoRAs to process\n")
    
    # Statistics
    images_downloaded = 0
    images_skipped = 0
    images_failed = 0
    loras_downloaded = 0
    loras_skipped = 0
    loras_failed = 0
    
    # Process each LoRA
    for idx, lora in enumerate(tqdm(loras, desc="Processing LoRAs"), 1):
        repo = lora.get("repo", "")
        title = lora.get("title", "Unknown")
        image_url = lora.get("image", "")
        weight_name = lora.get("weights", None)
        
        if not repo:
            print(f"\n⚠️  Entry {idx}: Missing repo, skipping")
            continue
        
        print(f"\n[{idx}/{total}] {title} ({repo})")
        
        # Download image
        if image_url:
            image_filename = get_image_filename(image_url, title)
            image_path = os.path.join(LORAS_IMAGES_DIR, image_filename)
            
            # Check if image already exists
            if os.path.exists(image_path):
                print(f"  ✓ Image already cached: {image_filename}")
                images_skipped += 1
            else:
                print(f"  📥 Downloading image...")
                result = download_image(image_url, image_path)
                if result:
                    print(f"  ✓ Image saved: {image_filename}")
                    images_downloaded += 1
                else:
                    images_failed += 1
        else:
            print(f"  ⚠️  No image URL provided")
        
        # Download LoRA
        print(f"  📥 Downloading LoRA model...")
        lora_path, was_cached = download_lora(repo, weight_name)
        if lora_path:
            if was_cached:
                print(f"  ✓ LoRA already cached")
                loras_skipped += 1
            else:
                print(f"  ✓ LoRA downloaded successfully")
                loras_downloaded += 1
        else:
            print(f"  ❌ LoRA download failed")
            loras_failed += 1
        
        # Small delay to avoid rate limiting
        time.sleep(0.5)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"Images:")
    print(f"  ✓ Downloaded: {images_downloaded}")
    print(f"  ⊘ Skipped (already cached): {images_skipped}")
    print(f"  ❌ Failed: {images_failed}")
    print(f"\nLoRAs:")
    print(f"  ✓ Downloaded: {loras_downloaded}")
    print(f"  ⊘ Skipped (already cached): {loras_skipped}")
    print(f"  ❌ Failed: {loras_failed}")
    print(f"\nTotal processed: {total}")
    print("=" * 60)
    print("\n✅ Download complete! Your LoRAs are now cached locally.")
    print(f"   Images: {LORAS_IMAGES_DIR}")
    print(f"   Models: {LORAS_CACHE_DIR}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

