---
title: FLUX LoRA DLC
emoji: 🥳
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.1.0
app_file: app.py
pinned: true
license: creativeml-openrail-m
short_description: 270+ Impressive LoRAs for Flux.1
---


![FLUX LoRA DLC](screen.png)

## Installation

### Prerequisites

1. **NVIDIA GPU with CUDA support** - This application requires CUDA
2. **Python 3.8+**
3. **NVIDIA GPU Drivers** - Install from [nvidia.com/drivers](https://www.nvidia.com/drivers)

### Setup Steps

1. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # or
   source venv/bin/activate  # Linux/Mac
   ```

2. **Install PyTorch with CUDA support:**
   
   **Option A - Using the provided script (Windows):**
   ```powershell
   .\install_cuda_pytorch.ps1
   ```
   or
   ```cmd
   install_cuda_pytorch.bat
   ```
   
   **Option B - Manual installation:**
   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Install other requirements:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify CUDA is working:**
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```
   Should output: `CUDA available: True`

### Troubleshooting

If CUDA is not available:

1. **Check NVIDIA GPU:**
   ```bash
   nvidia-smi
   ```
   If this fails, install/update NVIDIA drivers.

2. **Verify PyTorch CUDA build:**
   ```bash
   python -c "import torch; print(torch.version.cuda)"
   ```
   Should show a CUDA version (e.g., `12.1`), not `None`.

3. **Reinstall PyTorch with CUDA:**
   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Install CUDA Toolkit** (if needed):
   - Download from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
   - Restart computer after installation

### Running the Application

```bash
python app.py
```

The application will:
- Use local `models_cache` folder (not C drive `.cache`)
- Require CUDA to run
- Show detailed diagnostics on startup

---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 54.00 MiB. GPU 0 has a total capacity of 11.99 GiB of which 0 bytes is free. Of the allocated memory 26.40 GiB is allocated by PyTorch, and 8.13 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

(venv) PS D:\AI-projects\FLUX-LoRA-DLC> setx PYTORCH_CUDA_ALLOC_CONF "expandable_segments:True"