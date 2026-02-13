# You'll get the llama.cpp-master.7z in Pyduino Installer 1.4.0 [llama.cpp-master] one it would be labeled first run the installer then run the Pyduino installer then run the extract_llama.cpp_master_7z.py using
```cmd
python extract_llama.cpp_master_7z.py
```
# no need to install python the pyduino installer already installs for you if you dont have it then it would be extracted then follow this Markdown

-----------------------------------
------------------------------------------
------------------
-------------------

# Complete Guide: Building llama.cpp on Windows with Visual Studio 2022

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Understanding Your Hardware](#understanding-your-hardware)
- [Installation Methods](#installation-methods)
  - [Method 1: CPU-Only Build](#method-1-cpu-only-build)
  - [Method 2: CUDA (NVIDIA GPU) Build](#method-2-cuda-nvidia-gpu-build)
  - [Method 3: Vulkan (Cross-Platform GPU) Build](#method-3-vulkan-cross-platform-gpu-build)
  - [Method 4: OpenBLAS (CPU Optimization) Build](#method-4-openblas-cpu-optimization-build)
- [Verification and Testing](#verification-and-testing)
- [Troubleshooting](#troubleshooting)
- [Performance Optimization](#performance-optimization)
- [Additional Resources](#additional-resources)

---

## Overview

**What is llama.cpp?**

llama.cpp is a C/C++ implementation of Meta's LLaMA (Large Language Model Meta AI) inference engine. It allows you to run large language models locally on your hardware with minimal dependencies and excellent performance.

**Why build from source?**

- Access to the latest features and optimizations
- Custom hardware acceleration support
- Better understanding of the tool
- Ability to modify and extend functionality
- Optimized specifically for your hardware configuration

**What you'll achieve:**

By the end of this guide, you'll have a fully functional llama.cpp installation optimized for your specific hardware, whether that's CPU-only, NVIDIA GPU (CUDA), cross-platform GPU (Vulkan), or CPU-optimized (OpenBLAS).

---

## Prerequisites

### Required Software

1. **Visual Studio 2022** (Community Edition is free)
   - Download: https://visualstudio.microsoft.com/downloads/
   - **Required Components during installation:**
     - Desktop development with C++
     - Windows 10/11 SDK
     - MSVC v143 or later build tools
     - CMake tools for Windows

2. **Git for Windows**
   - Download: https://git-scm.com/download/win
   - Required to clone the llama.cpp repository

3. **CMake** (usually included with VS 2022, but standalone also available)
   - Download: https://cmake.org/download/
   - Version 3.15 or higher required

### Optional Software (depending on your build type)

4. **CUDA Toolkit** (for NVIDIA GPU support)
   - Download: https://developer.nvidia.com/cuda-downloads
   - Version 11.0 or higher recommended
   - Requires NVIDIA GPU with compute capability 6.0+

5. **Vulkan SDK** (for cross-platform GPU support)
   - Download: https://vulkan.lunarg.com/sdk/home
   - Works with NVIDIA, AMD, and Intel GPUs
   - Latest version recommended

6. **OpenBLAS** (for CPU optimization)
   - We'll build this as part of the process or use vcpkg
   - Provides optimized linear algebra operations

### System Requirements

**Minimum:**
- Windows 10/11 (64-bit)
- 8GB RAM
- 10GB free disk space
- Modern CPU (Intel/AMD from last 5 years)

**Recommended:**
- Windows 11 (64-bit)
- 16GB+ RAM
- 20GB+ free disk space (for models)
- Multi-core CPU or dedicated GPU

---

## Understanding Your Hardware

Before building, let's identify your hardware to choose the optimal build configuration.

### Check Your GPU

**Method 1: Task Manager**
1. Press `Ctrl + Shift + Esc`
2. Click "Performance" tab
3. Look for GPU section on the left

**Method 2: Command Line**
```cmd
wmic path win32_VideoController get name
```

**Method 3: DirectX Diagnostic**
1. Press `Win + R`
2. Type `dxdiag` and press Enter
3. Click "Display" tab

### GPU Type Decision Tree

```
Do you have a GPU?
│
├─ YES → What brand?
│   ├─ NVIDIA (GeForce/RTX/Quadro)
│   │   └─ Use CUDA build (best performance for NVIDIA)
│   │       Alternative: Vulkan (if CUDA has issues)
│   │
│   ├─ AMD (Radeon)
│   │   └─ Use Vulkan build (primary option for AMD)
│   │       Alternative: ROCm (advanced, not covered here)
│   │
│   └─ Intel (Arc/Iris)
│       └─ Use Vulkan build (works with modern Intel GPUs)
│
└─ NO or UNSURE → Use CPU-only build
    └─ Consider OpenBLAS for better CPU performance
```

### Check NVIDIA Compute Capability (NVIDIA Users Only)

```cmd
nvidia-smi --query-gpu=compute_cap --format=csv
```

Requirements:
- CUDA requires compute capability **6.0 or higher**
- Find your GPU: https://developer.nvidia.com/cuda-gpus

---

## Installation Methods

### Preparing Your Environment

**Step 1: Open x64 Native Tools Command Prompt for VS 2022**

This is crucial - you must use the correct command prompt!

**Option A: Through Start Menu**
1. Press Windows key
2. Type "x64 Native Tools"
3. Click "x64 Native Tools Command Prompt for VS 2022"

**Option B: Through Visual Studio**
1. Open Visual Studio 2022
2. Go to Tools → Command Line → Developer Command Prompt
3. This opens a command prompt window

**Option C: Manual Method**
```cmd
"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
```

**Verify your environment:**
```cmd
cl
```
You should see the Microsoft C/C++ Compiler version information.

```cmd
cmake --version
```
You should see CMake version 3.15 or higher.

**Step 2: Create a workspace directory**

```cmd
mkdir C:\LLaMa
cd C:\LLaMa
```

**Step 3: Clone the repository**

```cmd
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
```

**Step 4: Check the latest commit (optional)**

```cmd
git log -1
```

This shows you which version you're building. You can always update later with `git pull`.

---

## Method 1: CPU-Only Build

**Best for:**
- No dedicated GPU
- Testing/development
- Maximum compatibility
- Laptops with integrated graphics

### Build Steps

**Step 1: Create build directory**

```cmd
mkdir build
cd build
```

**Step 2: Configure with CMake**

```cmd
cmake .. -G "Visual Studio 17 2022" -A x64
```

**Explanation:**
- `..` - Parent directory (where CMakeLists.txt is located)
- `-G "Visual Studio 17 2022"` - Use Visual Studio 2022 generator
- `-A x64` - Target 64-bit architecture

**Expected output:**
```
-- The C compiler identification is MSVC 19.x.x
-- The CXX compiler identification is MSVC 19.x.x
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Configuring done
-- Generating done
-- Build files have been written to: C:/LLaMa/llama.cpp/build
```

**Step 3: Build the project**

```cmd
cmake --build . --config Release
```

**Explanation:**
- `--build .` - Build in current directory
- `--config Release` - Build optimized release version (not debug)

**Build time:** 2-5 minutes depending on your CPU

**Expected output:**
```
Microsoft (R) Build Engine version ...
Building Custom Rule ...
[100%] Built target llama-cli
```

**Step 4: Verify binaries**

```cmd
dir bin\Release
```

You should see:
- `llama-cli.exe` - Main inference executable
- `llama-server.exe` - Server mode
- `llama-embedding.exe` - Generate embeddings
- `llama-quantize.exe` - Quantize models
- Various other utilities

**Step 5: Test the build**

```cmd
bin\Release\llama-cli.exe --version
```


---

## Method 2: CUDA (NVIDIA GPU) Build

**Best for:**
- NVIDIA GeForce/RTX/Quadro GPUs
- Best performance on NVIDIA hardware
- Compute capability 6.0+ required

### Prerequisites Check

**Step 1: Verify CUDA installation**

```cmd
nvcc --version
```

Expected output:
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on ...
Cuda compilation tools, release 12.x, Vxx.x.xxx
```

If you don't see this, install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads

**Step 2: Verify GPU capability**

```cmd
nvidia-smi
```

Check your GPU model and note the name.

**Step 3: Check compute capability**

Visit https://developer.nvidia.com/cuda-gpus and find your GPU. Ensure compute capability is 6.0 or higher.

Common GPUs:
- GTX 1050/1060/1070/1080 series: 6.1
- RTX 2060/2070/2080 series: 7.5
- RTX 3060/3070/3080/3090 series: 8.6
- RTX 4060/4070/4080/4090 series: 8.9

### Build Steps

**Step 1: Clean previous builds (if any)**

```cmd
cd C:\LLaMa\llama.cpp
rmdir /s /q build
mkdir build
cd build
```

**Step 2: Configure with CUDA enabled**

```cmd
cmake .. -G "Visual Studio 17 2022" -A x64 -DGGML_CUDA=ON
```

**Explanation:**
- `-DGGML_CUDA=ON` - Enable CUDA support

**Advanced options (optional):**

For specific CUDA architecture targeting (better performance):
```cmd
cmake .. -G "Visual Studio 17 2022" -A x64 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86;89"
```

Common architectures:
- `61` - GTX 1000 series
- `75` - RTX 2000 series
- `86` - RTX 3000 series
- `89` - RTX 4000 series

**Expected CMake output:**
```
-- GGML CUDA sources found, configuring CUDA options
-- CUDA Architectures: 86;89
-- Using CUDA architectures: 86;89
...
-- Configuring done
-- Generating done
```

**Step 3: Build the project**

```cmd
cmake --build . --config Release
```

**Build time:** 5-15 minutes (CUDA compilation is slower)

**Step 4: Verify CUDA build**

```cmd
bin\Release\llama-cli.exe --version
```

Then run a quick check:
```cmd
bin\Release\llama-cli.exe -ngl 99 --version
```

The `-ngl 99` flag tells llama.cpp to offload 99 layers to GPU. If you see errors, CUDA may not be properly enabled.

### CUDA Environment Variables (Optional Optimization)

Add these to improve performance:

```cmd
set CUDA_VISIBLE_DEVICES=0
set CUDA_LAUNCH_BLOCKING=0
```

For multiple GPUs:
```cmd
set CUDA_VISIBLE_DEVICES=0,1
```

---

## Method 3: Vulkan (Cross-Platform GPU) Build

**Best for:**
- AMD GPUs (Radeon series)
- Intel Arc GPUs
- NVIDIA GPUs (alternative to CUDA)
- When you want portability
- Newer GPU architectures

### Prerequisites Check

**Step 1: Verify Vulkan SDK installation**

```cmd
where vulkaninfo
```

If not found, download and install Vulkan SDK from https://vulkan.lunarg.com/sdk/home

**Step 2: Verify Vulkan runtime**

```cmd
vulkaninfo --summary
```

Expected output should show your GPU and Vulkan version:
```
Vulkan Instance Version: 1.3.xxx

Instance Extensions:
...

GPU0:
    apiVersion         = 1.3.xxx
    driverVersion      = ...
    vendorID           = 0x10de (NVIDIA) / 0x1002 (AMD) / 0x8086 (Intel)
    deviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
    deviceName         = Your GPU Name
```

### Build Steps

**Step 1: Clean previous builds**

```cmd
cd C:\LLaMa\llama.cpp
rmdir /s /q build
mkdir build
cd build
```

**Step 2: Configure with Vulkan enabled**

```cmd
cmake .. -G "Visual Studio 17 2022" -A x64 -DGGML_VULKAN=ON
```

**Explanation:**
- `-DGGML_VULKAN=ON` - Enable Vulkan support

**Expected CMake output:**
```
-- Found Vulkan: C:/VulkanSDK/x.x.xxx.x/Lib/vulkan-1.lib
-- GGML Vulkan support enabled
...
-- Configuring done
-- Generating done
```

**Step 3: Build the project**

```cmd
cmake --build . --config Release
```

**Build time:** 3-8 minutes

**Step 4: Verify Vulkan build**

```cmd
bin\Release\llama-cli.exe --version
```

Check GPU detection:
```cmd
set GGML_VULKAN_DEBUG=1
bin\Release\llama-cli.exe -ngl 99 --version
```

You should see output about Vulkan initialization and your GPU being detected.

### Vulkan Troubleshooting

**If GPU is not detected:**

1. Check Vulkan devices:
```cmd
set GGML_VULKAN_DEBUG=1
set GGML_VULKAN_DEVICE=0
```

2. List available devices:
```cmd
vulkaninfo --summary
```

3. Force specific device (if you have multiple GPUs):
```cmd
set GGML_VULKAN_DEVICE=1
```

---

## Method 4: OpenBLAS (CPU Optimization) Build

**Best for:**
- CPU-only systems needing better performance
- Multi-core CPUs
- When GPU acceleration is unavailable
- Servers without GPUs

### Prerequisites: Install OpenBLAS via vcpkg

**Step 1: Install vcpkg (if not already installed)**

```cmd
cd C:\
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
bootstrap-vcpkg.bat
```

**Step 2: Install OpenBLAS**

```cmd
vcpkg install openblas:x64-windows
```

This will take 5-15 minutes to download and build OpenBLAS.

**Step 3: Integrate vcpkg with Visual Studio**

```cmd
vcpkg integrate install
```

### Build Steps

**Step 1: Set vcpkg toolchain**

```cmd
cd C:\LLaMa\llama.cpp
rmdir /s /q build
mkdir build
cd build
```

**Step 2: Configure with OpenBLAS**

```cmd
cmake .. -G "Visual Studio 17 2022" -A x64 -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
```

**Explanation:**
- `-DGGML_BLAS=ON` - Enable BLAS support
- `-DGGML_BLAS_VENDOR=OpenBLAS` - Use OpenBLAS specifically
- `-DCMAKE_TOOLCHAIN_FILE=...` - Tell CMake where to find vcpkg packages

**Expected output:**
```
-- Found OpenBLAS: C:/vcpkg/installed/x64-windows/lib/openblas.lib
-- GGML BLAS enabled with OpenBLAS
...
-- Configuring done
```

**Step 3: Build the project**

```cmd
cmake --build . --config Release
```

**Step 4: Copy OpenBLAS DLL**

```cmd
copy C:\vcpkg\installed\x64-windows\bin\openblas.dll bin\Release\
```

**Important:** The OpenBLAS DLL must be in the same directory as the executable or in your PATH.

**Step 5: Verify OpenBLAS build**

```cmd
bin\Release\llama-cli.exe --version
```

---

## Advanced: Combining Multiple Backends

You can enable multiple backends simultaneously!

### CUDA + OpenBLAS

```cmd
cmake .. -G "Visual Studio 17 2022" -A x64 ^
    -DGGML_CUDA=ON ^
    -DGGML_BLAS=ON ^
    -DGGML_BLAS_VENDOR=OpenBLAS ^
    -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
```

### Vulkan + OpenBLAS

```cmd
cmake .. -G "Visual Studio 17 2022" -A x64 ^
    -DGGML_VULKAN=ON ^
    -DGGML_BLAS=ON ^
    -DGGML_BLAS_VENDOR=OpenBLAS ^
    -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
```

---

## Verification and Testing

### Download a Test Model

We'll use a small model for testing.

**Step 1: Create models directory**

```cmd
cd C:\LLaMa\llama.cpp
mkdir models
cd models
```

**Step 2: Download a small GGUF model**

Using PowerShell (open PowerShell):
```powershell
Invoke-WebRequest -Uri "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" -OutFile "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
```

Or use curl (if available):
```cmd
curl -L "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" -o tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

Or manually download from HuggingFace and place in the models folder.

### Test Basic Inference

**CPU-only test:**

```cmd
cd C:\LLaMa\llama.cpp
bin\Release\llama-cli.exe -m models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -p "Hello, my name is" -n 50
```

**Parameters explained:**
- `-m` - Model file path
- `-p` - Prompt text
- `-n 50` - Generate 50 tokens

**GPU test (CUDA/Vulkan):**

```cmd
bin\Release\llama-cli.exe -m models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -p "Hello, my name is" -n 50 -ngl 99
```

**Parameters explained:**
- `-ngl 99` - Offload 99 layers to GPU (adjust based on your VRAM)

### Interactive Mode

Test chat functionality:

```cmd
bin\Release\llama-cli.exe -m models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --interactive --color
```


Type messages and press Enter. Type `/exit` to quit.

### Performance Benchmarking

**Prompt processing benchmark:**

```cmd
bin\Release\llama-bench.exe -m models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

This shows:
- Tokens per second (t/s) for prompt processing
- Tokens per second for generation
- Memory usage

**Example output:**
```
| model                          |       size |     params | backend    | ngl | test       |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------- | ---------------: |
| tinyllama 1B Q4_K - Medium     |   638.00 MB |     1.10 B | CUDA       |  99 | pp512      |   1234.56 ± 5.67 |
| tinyllama 1B Q4_K - Medium     |   638.00 MB |     1.10 B | CUDA       |  99 | tg128      |    123.45 ± 1.23 |
```

**Interpret results:**
- `pp512` - Prompt processing (512 tokens)
- `tg128` - Text generation (128 tokens)
- Higher t/s = better performance

---

## Troubleshooting

### Common Build Errors

#### Error: "CMake is not recognized"

**Problem:** CMake not in PATH

**Solution:**
```cmd
set PATH=%PATH%;C:\Program Files\CMake\bin
```

Or reinstall Visual Studio 2022 with CMake tools checked.

#### Error: "CUDA toolkit not found"

**Problem:** CUDA not installed or not in PATH

**Solution 1: Add CUDA to PATH**
```cmd
set PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x
```

**Solution 2: Reinstall CUDA Toolkit**

Download from https://developer.nvidia.com/cuda-downloads and ensure you check "Add to PATH" during installation.

#### Error: "Vulkan SDK not found"

**Problem:** Vulkan not installed

**Solution:**
1. Download from https://vulkan.lunarg.com/sdk/home
2. Install with default options
3. Restart command prompt
4. Verify: `where vulkaninfo`

#### Error: "MSBuild failed with errors"

**Problem:** Visual Studio components missing

**Solution:**

1. Open Visual Studio Installer
2. Modify Visual Studio 2022
3. Ensure these are checked:
   - Desktop development with C++
   - Windows 10/11 SDK
   - MSVC v143 build tools
   - C++ CMake tools for Windows
4. Click Modify and wait for installation

#### Error: "LNK2001: unresolved external symbol"

**Problem:** Missing libraries or incorrect configuration

**Solution for OpenBLAS:**
```cmd
# Ensure OpenBLAS DLL is copied
copy C:\vcpkg\installed\x64-windows\bin\openblas.dll bin\Release\

# Verify vcpkg integration
vcpkg integrate install
```

**Solution for CUDA:**
```cmd
# Verify CUDA environment variables
echo %CUDA_PATH%
# Should show: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x
```

#### Error: "git is not recognized"

**Problem:** Git not installed or not in PATH

**Solution:**
Download Git for Windows: https://git-scm.com/download/win

Or use winget:
```cmd
winget install --id Git.Git -e --source winget
```

#### Error: "fatal: not a git repository"

**Problem:** Not in the right directory

**Solution:**
```cmd
cd C:\LLaMa\llama.cpp
git status
```

#### Error: Out of Memory during build

**Problem:** Not enough RAM for parallel build

**Solution: Limit parallel jobs**
```cmd
cmake --build . --config Release -- /m:2
```

The `/m:2` limits to 2 parallel jobs. Adjust based on your RAM (4GB RAM = /m:1, 8GB RAM = /m:2, 16GB+ = /m:4).

### Runtime Errors

#### Error: "The code execution cannot proceed because VCRUNTIME140.dll was not found"

**Problem:** Missing Visual C++ Redistributables

**Solution:**

Download and install Microsoft Visual C++ Redistributable:
https://aka.ms/vs/17/release/vc_redist.x64.exe

#### Error: "CUDA out of memory"

**Problem:** Model too large for GPU VRAM

**Solution: Reduce GPU layers**
```cmd
# Instead of -ngl 99, use fewer layers
bin\Release\llama-cli.exe -m model.gguf -p "test" -ngl 20
```

Start with `-ngl 10` and increase until you hit memory limit.

**Check VRAM usage:**
```cmd
nvidia-smi
```

Look at "Memory-Usage" column.

#### Error: "ggml_init_cublas: GGML_CUDA_FORCE_MMQ is deprecated"

**Problem:** Using deprecated environment variable

**Solution: Remove old CUDA settings**
```cmd
set GGML_CUDA_FORCE_MMQ=
set GGML_CUDA_FORCE_DMMV=
```

These are deprecated. Modern llama.cpp auto-selects best kernels.

#### Error: "Illegal memory access" with CUDA

**Problem:** GPU driver or CUDA version mismatch

**Solution:**

1. Update NVIDIA drivers:
   - Visit https://www.nvidia.com/Download/index.aspx
   - Download latest Game Ready or Studio driver

2. Check CUDA compatibility:
   ```cmd
   nvidia-smi
   ```
   Note "CUDA Version" on top right. Your CUDA Toolkit should be ≤ this version.

3. Rebuild with different CUDA version if needed

#### Error: Vulkan device not found

**Problem:** GPU not Vulkan-compatible or drivers outdated

**Solution:**

1. Update GPU drivers
2. Check Vulkan support:
   ```cmd
   vulkaninfo --summary
   ```

3. If no devices shown, your GPU may not support Vulkan
   - Minimum: NVIDIA 600 series, AMD GCN 1.0, Intel HD 4000

---

## Performance Optimization

### CPU Optimization

**1. Thread count optimization**

```cmd
# Auto-detect optimal threads (recommended)
bin\Release\llama-cli.exe -m model.gguf -p "test" -t 0

# Manual thread count (number of physical cores)
bin\Release\llama-cli.exe -m model.gguf -p "test" -t 8
```

Find your core count:
```cmd
wmic cpu get NumberOfCores,NumberOfLogicalProcessors
```

**Best practice:** Use physical cores, not logical (hyperthreading) cores.

**2. Batch size tuning**

```cmd
# Larger batch = better throughput, more memory
bin\Release\llama-cli.exe -m model.gguf -p "test" -b 512

# Smaller batch = less memory, slower
bin\Release\llama-cli.exe -m model.gguf -p "test" -b 128
```

Default is 512. Reduce if you run out of RAM.

**3. Memory locking (reduce page faults)**

```cmd
bin\Release\llama-cli.exe -m model.gguf -p "test" --mlock
```

Keeps model in RAM, prevents swapping to disk. Requires enough physical RAM.

### GPU Optimization

**1. Layer offloading strategy**

```cmd
# Check model layer count
bin\Release\llama-cli.exe -m model.gguf --verbose

# Offload all layers (if VRAM allows)
-ngl 99

# Partial offload (balance VRAM/RAM)
-ngl 30
```

**Rule of thumb:**
- 8GB VRAM: -ngl 20-30 for 7B models
- 12GB VRAM: -ngl 35-45 for 7B models
- 16GB+ VRAM: -ngl 99 (all layers)

**2. Batch size for GPU**

```cmd
# Larger batches utilize GPU better
bin\Release\llama-cli.exe -m model.gguf -p "test" -ngl 99 -b 512

# For very large batches (server mode)
bin\Release\llama-cli.exe -m model.gguf -p "test" -ngl 99 -b 2048
```

**3. Context size optimization**

```cmd
# Default context (2048 tokens)
bin\Release\llama-cli.exe -m model.gguf -p "test" -ngl 99

# Larger context (more memory)
bin\Release\llama-cli.exe -m model.gguf -p "test" -ngl 99 -c 4096

# Smaller context (save VRAM)
bin\Release\llama-cli.exe -m model.gguf -p "test" -ngl 99 -c 1024
```

**Memory usage formula (approximate):**
- VRAM needed ≈ Model size + (context × layers × 2 bytes)

**4. Flash Attention (modern GPUs)**

Enabled automatically on compatible GPUs (CUDA compute 7.0+). Check with:

```cmd
bin\Release\llama-cli.exe -m model.gguf --verbose
```

Look for "flash_attn" in output.

### Model Quantization for Performance

Smaller quantized models run faster:

**Quantization levels (from largest to smallest):**
- F16 - Full precision (largest, slowest)
- Q8_0 - 8-bit (excellent quality, good speed)
- Q6_K - 6-bit (great quality, faster)
- Q5_K_M - 5-bit medium (good quality, fast)
- Q4_K_M - 4-bit medium (decent quality, very fast) **← RECOMMENDED**
- Q3_K_M - 3-bit medium (lower quality, very fast)
- Q2_K - 2-bit (poor quality, extremely fast)

**Example download URLs (replace MODEL with actual model name):**
```
https://huggingface.co/TheBloke/MODEL-GGUF/resolve/main/model-name.Q4_K_M.gguf
https://huggingface.co/TheBloke/MODEL-GGUF/resolve/main/model-name.Q5_K_M.gguf
```

**Quantize your own models:**

```cmd
# Convert from PyTorch/SafeTensors to F16 GGUF
python convert_hf_to_gguf.py path\to\model --outfile model-f16.gguf

# Quantize to Q4_K_M
bin\Release\llama-quantize.exe model-f16.gguf model-q4_k_m.gguf Q4_K_M
```

---

## Environment Variables Reference

### CUDA Variables

```cmd
# Force specific GPU (multi-GPU systems)
set CUDA_VISIBLE_DEVICES=0

# Enable CUDA graphs (may improve performance)
set GGML_CUDA_ENABLE_GRAPH=1

# Set number of streams (advanced)
set GGML_CUDA_STREAMS=4
```

### Vulkan Variables

```cmd
# Force specific device
set GGML_VULKAN_DEVICE=0

# Enable debug output
set GGML_VULKAN_DEBUG=1

# Limit memory allocation
set GGML_VULKAN_MAX_ALLOC=8192
```

### General Variables

```cmd
# Set thread count globally
set OMP_NUM_THREADS=8

# Disable memory locking warning
set GGML_NO_MLOCK_WARNING=1

# Enable performance logging
set LLAMA_DEBUG=1
```

---

## Adding to PATH (Optional but Recommended)

To run llama.cpp from any directory:

**Temporary (current session only):**
```cmd
set PATH=%PATH%;C:\LLaMa\llama.cpp\build\bin\Release
```

**Permanent (requires admin):**

1. Press `Win + X`, select "System"
2. Click "Advanced system settings"
3. Click "Environment Variables"
4. Under "User variables" or "System variables", find "Path"
5. Click "Edit"
6. Click "New"
7. Add: `C:\LLaMa\llama.cpp\build\bin\Release`
8. Click OK on all dialogs
9. **Restart command prompt**

Now you can run:
```cmd
llama-cli.exe -m path\to\model.gguf -p "test"
```

From any directory!

---

## Server Mode

Run llama.cpp as an OpenAI-compatible API server:

```cmd
bin\Release\llama-server.exe -m models\model.gguf -ngl 99 --host 127.0.0.1 --port 8080
```

**Parameters:**
- `--host 127.0.0.1` - Listen on localhost only (safe)
- `--port 8080` - Port number
- `-c 4096` - Context window size
- `-ngl 99` - GPU layers

**Access the web UI:**
Open browser: `http://localhost:8080`

**API endpoint:**
```
POST http://localhost:8080/v1/chat/completions
```

Compatible with OpenAI Python client!


---

## Using Visual Studio Code (Alternative to Command Line)

You can also build llama.cpp using Visual Studio Code with CMake Tools extension.

### Setup Visual Studio Code

**Step 1: Install VS Code**
Download from https://code.visualstudio.com/

**Step 2: Install Extensions**

Required extensions:
1. **C/C++** (Microsoft)
2. **CMake Tools** (Microsoft)
3. **CMake** (twxs)

Install via Extensions panel (Ctrl+Shift+X) or:
```cmd
code --install-extension ms-vscode.cpptools
code --install-extension ms-vscode.cmake-tools
code --install-extension twxs.cmake
```

**Step 3: Open llama.cpp in VS Code**

```cmd
cd C:\LLaMa\llama.cpp
code .
```

### Configure CMake in VS Code

**Step 1: Select Kit**

1. Press `Ctrl+Shift+P`
2. Type "CMake: Select a Kit"
3. Choose "Visual Studio Community 2022 Release - amd64"

**Step 2: Configure CMake**

1. Press `Ctrl+Shift+P`
2. Type "CMake: Configure"
3. Wait for configuration to complete

**For CUDA build:**

Edit `.vscode/settings.json` (create if doesn't exist):
```json
{
    "cmake.configureArgs": [
        "-DGGML_CUDA=ON"
    ]
}
```

**For Vulkan build:**
```json
{
    "cmake.configureArgs": [
        "-DGGML_VULKAN=ON"
    ]
}
```

**For OpenBLAS build:**
```json
{
    "cmake.configureArgs": [
        "-DGGML_BLAS=ON",
        "-DGGML_BLAS_VENDOR=OpenBLAS",
        "-DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake"
    ]
}
```

**Step 3: Build**

1. Press `Ctrl+Shift+P`
2. Type "CMake: Build"
3. Or press `F7`
4. Or click "Build" in the status bar

**Step 4: Run/Debug**

1. Press `Ctrl+Shift+P`
2. Type "CMake: Run Without Debugging"
3. Or press `Shift+F5`

### VS Code CMake Shortcuts

- `Ctrl+Shift+P` → `CMake: Configure` - Configure project
- `F7` - Build
- `Shift+F5` - Run without debugging
- `F5` - Debug
- `Ctrl+Shift+P` → `CMake: Clean` - Clean build
- `Ctrl+Shift+P` → `CMake: Delete Cache and Reconfigure` - Start fresh

### Debugging in VS Code

**Step 1: Create launch configuration**

Create `.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "llama-cli (Debug)",
            "type": "cppvsdbg",
            "request": "launch",
            "program": "${workspaceFolder}/build/bin/Debug/llama-cli.exe",
            "args": [
                "-m", "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
                "-p", "Hello",
                "-n", "50"
            ],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "environment": [],
            "console": "integratedTerminal"
        }
    ]
}
```

**Step 2: Set breakpoints**

1. Open source file (e.g., `examples/main/main.cpp`)
2. Click left of line number to set breakpoint (red dot)

**Step 3: Start debugging**

1. Press `F5`
2. Use debug toolbar to step through code

---

## Keeping llama.cpp Updated

### Update to Latest Version

**Step 1: Navigate to repository**
```cmd
cd C:\LLaMa\llama.cpp
```

**Step 2: Save any local changes (if you modified code)**
```cmd
git stash
```

**Step 3: Pull latest changes**
```cmd
git pull origin master
```

**Step 4: Clean and rebuild**
```cmd
cd build
cmake --build . --config Release --clean-first
```

Or completely rebuild:
```cmd
rmdir /s /q build
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -DGGML_CUDA=ON
cmake --build . --config Release
```

### Check for Updates

```cmd
cd C:\LLaMa\llama.cpp
git fetch origin
git log HEAD..origin/master --oneline
```

Shows commits you're missing.

### Subscribe to Updates

Star the repository on GitHub to get notifications:
https://github.com/ggerganov/llama.cpp

---

## Working with Models

### Where to Find Models

**Popular GGUF repositories:**

1. **TheBloke (most popular)**
   - https://huggingface.co/TheBloke
   - Hundreds of quantized models
   - Example: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF

2. **Official model repos**
   - https://huggingface.co/meta-llama
   - https://huggingface.co/mistralai

3. **Community models**
   - Browse https://huggingface.co/models?library=gguf

### Download Models

**Using browser:**
1. Go to model page on HuggingFace
2. Click "Files and versions"
3. Download `.gguf` file
4. Move to `C:\LLaMa\llama.cpp\models\`

**Using command line (PowerShell):**
```powershell
# Small 1B model
Invoke-WebRequest -Uri "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" -OutFile "models\tinyllama.gguf"

# 7B model (4.4GB)
Invoke-WebRequest -Uri "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf" -OutFile "models\llama-2-7b.gguf"
```

**Using huggingface-cli (recommended for large models):**

```cmd
pip install huggingface-hub

huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF llama-2-7b-chat.Q4_K_M.gguf --local-dir models --local-dir-use-symlinks False
```

### Model Size vs VRAM Requirements

**Approximate VRAM needed (with -ngl 99, full GPU offload):**

| Model Size | Quantization | VRAM Required |
|------------|--------------|---------------|
| 1B         | Q4_K_M       | 1-2 GB        |
| 3B         | Q4_K_M       | 2-3 GB        |
| 7B         | Q4_K_M       | 4-5 GB        |
| 7B         | Q5_K_M       | 5-6 GB        |
| 13B        | Q4_K_M       | 8-9 GB        |
| 34B        | Q4_K_M       | 20-22 GB      |
| 70B        | Q4_K_M       | 40-45 GB      |

**For CPU-only (RAM requirements):**
Similar to VRAM requirements but use system RAM.

### Recommended Models for Beginners

**Very small (testing, learning):**
- TinyLlama-1.1B-Chat (600MB)
- Phi-2 (1.7GB)

**Small (daily use, fast):**
- Mistral-7B-Instruct (4.4GB)
- Llama-2-7B-Chat (4.4GB)

**Medium (better quality):**
- Mixtral-8x7B-Instruct (26GB)
- Yi-34B-Chat (20GB)

**Large (best quality, slow):**
- Llama-2-70B-Chat (40GB)

---

## Advanced Topics

### Multi-GPU Support (CUDA)

**Split model across GPUs:**

```cmd
set CUDA_VISIBLE_DEVICES=0,1
bin\Release\llama-cli.exe -m model.gguf -ngl 99 -ts 20,10
```

**Parameters:**
- `-ts 20,10` - Tensor split: 20/30 on GPU0, 10/30 on GPU1
- Adjust ratios based on VRAM (e.g., `-ts 1,1` for equal split)

**Check GPU usage:**
```cmd
nvidia-smi dmon -s u -c 10
```

### Compiling for Different CPUs

**For AMD Ryzen (AVX2):**
```cmd
cmake .. -G "Visual Studio 17 2022" -A x64 -DGGML_AVX2=ON
```

**For older CPUs (no AVX):**
```cmd
cmake .. -G "Visual Studio 17 2022" -A x64 -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF
```

**For newest CPUs (AVX512):**
```cmd
cmake .. -G "Visual Studio 17 2022" -A x64 -DGGML_AVX512=ON
```

### Custom Build Options

**Enable all optimizations:**
```cmd
cmake .. -G "Visual Studio 17 2022" -A x64 ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DGGML_CUDA=ON ^
    -DGGML_BLAS=ON ^
    -DGGML_BLAS_VENDOR=OpenBLAS ^
    -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake ^
    -DGGML_NATIVE=ON
```

**Build specific targets only:**
```cmd
cmake --build . --config Release --target llama-cli
cmake --build . --config Release --target llama-server
```

---

## Batch Processing Scripts

### PowerShell Script for Batch Inference

Create `batch_inference.ps1`:

```powershell
# Batch inference script
$modelPath = "models\llama-2-7b-chat.Q4_K_M.gguf"
$prompts = @(
    "Explain quantum computing in simple terms.",
    "Write a haiku about programming.",
    "What is the capital of France?"
)

foreach ($prompt in $prompts) {
    Write-Host "Processing: $prompt" -ForegroundColor Green
    & "bin\Release\llama-cli.exe" -m $modelPath -p $prompt -n 100 -ngl 99
    Write-Host "`n---`n"
}
```

Run:
```cmd
powershell -ExecutionPolicy Bypass -File batch_inference.ps1
```

### Batch File for Quick Testing

Create `test_model.bat`:

```batch
@echo off
set MODEL=models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
set LAYERS=99

echo Testing model: %MODEL%
echo.

bin\Release\llama-cli.exe -m %MODEL% -p "Hello, how are you?" -n 50 -ngl %LAYERS%

pause
```

Run:
```cmd
test_model.bat
```

---

## Performance Monitoring

### Built-in Benchmarking

**Full benchmark:**
```cmd
bin\Release\llama-bench.exe -m models\model.gguf -p 512 -n 128 -ngl 99
```

**Parameters:**
- `-p 512` - Prompt size (tokens)
- `-n 128` - Generation size (tokens)
- `-ngl 99` - GPU layers

**Output explained:**
```
| model          | size    | params | backend | ngl | test  |         t/s |
|----------------|---------|--------|---------|-----|-------|-------------|
| llama 7B Q4_K  | 4.37 GB | 7.00 B | CUDA    | 99  | pp512 | 1234.56     |
| llama 7B Q4_K  | 4.37 GB | 7.00 B | CUDA    | 99  | tg128 | 98.76       |
```

- `pp512` - Prompt processing speed (tokens/sec)
- `tg128` - Text generation speed (tokens/sec)

### GPU Monitoring During Inference

**NVIDIA:**
```cmd
# Terminal 1: Run inference
bin\Release\llama-cli.exe -m model.gguf -p "test" -ngl 99

# Terminal 2: Monitor GPU
nvidia-smi dmon -s umct -d 1
```

**Watch specific metrics:**
```cmd
nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used --format=csv -l 1
```

### CPU Monitoring

```cmd
wmic cpu get loadpercentage
```

Or use Task Manager (Ctrl+Shift+Esc) → Performance tab.

---

## Security Considerations

### Safe Model Usage

**Warning:** Models downloaded from the internet can potentially contain malicious code if they're not in GGUF format or have been modified.

**Best practices:**
1. Only download from trusted sources (HuggingFace official repos)
2. Verify file hashes when provided
3. Use `.gguf` format (not pickle files)
4. Scan downloaded files with antivirus

### Network Security (Server Mode)

**Safe configuration (localhost only):**
```cmd
bin\Release\llama-server.exe -m model.gguf --host 127.0.0.1 --port 8080
```

**Dangerous configuration (exposed to network):**
```cmd
# Don't do this unless you know what you're doing!
bin\Release\llama-server.exe -m model.gguf --host 0.0.0.0 --port 8080
```

If you need network access, use a reverse proxy (nginx) with authentication.


---

## Integration Examples

### Python Integration

**Using llama-cpp-python (recommended):**

```cmd
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

**For CUDA 12.1 (adjust cu121 to your CUDA version)**

**Example usage:**
```python
from llama_cpp import Llama

# Load model
llm = Llama(
    model_path="models/llama-2-7b-chat.Q4_K_M.gguf",
    n_gpu_layers=99,  # Adjust based on your VRAM
    n_ctx=2048,       # Context window
)

# Generate
output = llm(
    "Q: What is the capital of France? A:",
    max_tokens=50,
    temperature=0.7,
)

print(output['choices'][0]['text'])
```

**Using subprocess (direct exe call):**
```python
import subprocess

def run_llama(prompt, model_path, n_tokens=50):
    cmd = [
        "bin/Release/llama-cli.exe",
        "-m", model_path,
        "-p", prompt,
        "-n", str(n_tokens),
        "-ngl", "99",
    ]
    
    result = subprocess.run(
        cmd, 
        capture_output=True, 
        text=True,
        cwd="C:/LLaMa/llama.cpp"
    )
    
    return result.stdout

# Use it
output = run_llama(
    "Explain AI in simple terms.",
    "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    100
)
print(output)
```

### Node.js Integration

**Install node-llama-cpp:**
```cmd
npm install node-llama-cpp
```

**Example usage:**
```javascript
const { LlamaCpp } = require('node-llama-cpp');

async function main() {
    const llama = new LlamaCpp({
        modelPath: 'models/llama-2-7b-chat.Q4_K_M.gguf',
        gpuLayers: 99,
    });
    
    const response = await llama.generate({
        prompt: 'What is JavaScript?',
        maxTokens: 100,
    });
    
    console.log(response);
}

main();
```

### REST API Client (Python)

```python
import requests

# Assuming llama-server is running on localhost:8080
url = "http://localhost:8080/v1/chat/completions"

payload = {
    "messages": [
        {"role": "user", "content": "Hello! What can you help me with?"}
    ],
    "temperature": 0.7,
    "max_tokens": 100,
}

response = requests.post(url, json=payload)
print(response.json()['choices'][0]['message']['content'])
```

### OpenAI Python Client

```python
from openai import OpenAI

# Point to local llama-server
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"  # llama-server doesn't require auth by default
)

completion = client.chat.completions.create(
    model="local-model",  # Model name doesn't matter for llama-server
    messages=[
        {"role": "user", "content": "Write a haiku about programming"}
    ],
    temperature=0.7,
)

print(completion.choices[0].message.content)
```

---

## Common Use Cases

### Chat Applications

**Interactive chat with system prompt:**

```cmd
bin\Release\llama-cli.exe ^
    -m models\llama-2-7b-chat.Q4_K_M.gguf ^
    -ngl 99 ^
    --interactive-first ^
    -r "User:" ^
    --in-prefix " " ^
    -p "System: You are a helpful AI assistant. Always be polite and concise.\n\nUser: "
```

**Explanation:**
- `--interactive-first` - Start in interactive mode
- `-r "User:"` - Reverse prompt (stops generation at this string)
- `--in-prefix " "` - Prefix before user input
- `-p` - Initial system prompt

### Code Generation

```cmd
bin\Release\llama-cli.exe ^
    -m models\codellama-7b.Q4_K_M.gguf ^
    -ngl 99 ^
    -p "Write a Python function to calculate fibonacci numbers:\n\n```python\n" ^
    -n 300 ^
    --temp 0.1
```

**Tips:**
- Use low temperature (0.1-0.3) for code
- Code-specific models work better (CodeLlama, WizardCoder)
- Use `\n\n```python\n` to prime for code output

### Document Summarization

```cmd
bin\Release\llama-cli.exe ^
    -m models\llama-2-7b-chat.Q4_K_M.gguf ^
    -ngl 99 ^
    -f document.txt ^
    -p "Summarize the following document in 3 bullet points:\n\n" ^
    -n 200
```

**Explanation:**
- `-f document.txt` - Load prompt from file
- Prefix sets the task

### Embeddings Generation

```cmd
bin\Release\llama-embedding.exe ^
    -m models\llama-2-7b.Q4_K_M.gguf ^
    -p "Text to embed" ^
    --embd-output-format json ^
    > embedding.json
```

**Use cases:**
- Semantic search
- Document similarity
- Clustering

### Batch File Processing

```cmd
for %%f in (*.txt) do (
    echo Processing %%f
    bin\Release\llama-cli.exe ^
        -m models\model.gguf ^
        -f "%%f" ^
        -p "Summarize: " ^
        -n 100 ^
        > "%%f.summary.txt"
)
```

---

## Recommended Configurations

### Configuration Files

Create reusable configs for common tasks.

**chat_config.txt:**
```
-m models/llama-2-7b-chat.Q4_K_M.gguf
-ngl 99
-c 4096
--interactive
--color
-r "User:"
--in-prefix " "
```

**Usage:**
```cmd
bin\Release\llama-cli.exe @chat_config.txt
```

### Preset Scripts

**Create `chat.bat`:**
```batch
@echo off
bin\Release\llama-cli.exe ^
    -m models\llama-2-7b-chat.Q4_K_M.gguf ^
    -ngl 99 ^
    -c 4096 ^
    --interactive ^
    --color ^
    -r "User:" ^
    --in-prefix " "
```

**Create `code.bat`:**
```batch
@echo off
bin\Release\llama-cli.exe ^
    -m models\codellama-7b.Q4_K_M.gguf ^
    -ngl 99 ^
    -c 8192 ^
    --interactive ^
    --temp 0.1 ^
    -r "Human:" ^
    -p "System: You are an expert programmer. Provide clean, well-documented code.\n\n"
```

Run with:
```cmd
chat.bat
code.bat
```

---

## FAQ (Frequently Asked Questions)

### Q: Which build should I use?

**A:** Decision tree:
1. Have NVIDIA GPU? → Use CUDA build (best performance)
2. Have AMD/Intel GPU? → Use Vulkan build
3. CPU only? → Use OpenBLAS build for better performance
4. Just testing? → Use basic CPU build

### Q: How much VRAM do I need?

**A:** Depends on model size and quantization:
- 7B Q4_K_M: ~5GB VRAM (with -ngl 99)
- 13B Q4_K_M: ~9GB VRAM
- 70B Q4_K_M: ~45GB VRAM

Use `-ngl` to adjust layer offloading if you don't have enough VRAM.

### Q: Why is generation slow?

**A:** Common causes:
1. No GPU acceleration (-ngl 0 or missing CUDA/Vulkan)
2. Large model for your hardware
3. Large context size (-c)
4. CPU-only with no OpenBLAS
5. Swapping to disk (not enough RAM/VRAM)

**Solutions:**
- Use GPU acceleration (-ngl 99)
- Use smaller/more quantized model (Q4_K_M instead of Q8)
- Reduce context (-c 2048 instead of 4096)
- Enable OpenBLAS for CPU
- Add more RAM/VRAM or use partial offloading

### Q: Can I use multiple GPUs?

**A:** Yes, with CUDA using `-ts` (tensor split):
```cmd
set CUDA_VISIBLE_DEVICES=0,1
llama-cli.exe -m model.gguf -ngl 99 -ts 1,1
```

Vulkan multi-GPU support is experimental.

### Q: What's the best quantization?

**A:** Depends on use case:
- **Q4_K_M**: Best balance (recommended for most)
- **Q5_K_M**: Better quality, slightly larger
- **Q8_0**: Near-original quality, much larger
- **Q3_K_M**: Smaller, faster, quality loss noticeable
- **Q2_K**: Very small, significant quality loss

### Q: How do I convert HuggingFace models?

**A:**
```cmd
# Install dependencies
pip install -r requirements.txt

# Convert
python convert_hf_to_gguf.py path\to\huggingface\model

# Quantize (optional)
bin\Release\llama-quantize.exe model.gguf model-q4.gguf Q4_K_M
```

### Q: Can I use models from other sources?

**A:** Yes! llama.cpp supports many model architectures:
- LLaMA/LLaMA 2/LLaMA 3
- Mistral/Mixtral
- Falcon
- GPT-2/GPT-J/GPT-NeoX
- StarCoder
- Many others

Check: https://github.com/ggerganov/llama.cpp#description

### Q: Is commercial use allowed?

**A:** llama.cpp itself is MIT licensed (permissive).

However, **model licenses vary**:
- LLaMA 2: Free for commercial use (with restrictions)
- LLaMA 3: Check Meta's license
- Mistral: Apache 2.0 (permissive)
- Many others: Check individual licenses

Always verify model license before commercial use!

### Q: How do I update llama.cpp?

**A:** See [Keeping llama.cpp Updated](#keeping-llamacpp-updated) section above.

Short version:
```cmd
cd C:\LLaMa\llama.cpp
git pull
cd build
cmake --build . --config Release --clean-first
```

### Q: What if I get "out of memory" errors?

**A:** Solutions:
1. Reduce GPU layers: `-ngl 20` instead of `-ngl 99`
2. Reduce context: `-c 2048` instead of `-c 4096`
3. Use more quantized model: Q4_K_M instead of Q5_K_M
4. Reduce batch size: `-b 256` instead of `-b 512`
5. Close other applications
6. Use smaller model (7B instead of 13B)

### Q: Can I run this on a laptop?

**A:** Yes! Options:
1. CPU-only with small models (1-3B)
2. Use laptop GPU (NVIDIA/AMD) with Vulkan/CUDA
3. Use quantized models (Q4_K_M or lower)
4. Reduce context size for less memory

Example for laptop:
```cmd
llama-cli.exe -m models\tinyllama.gguf -c 1024 -ngl 20
```

---

## Additional Resources

### Official Documentation

- **llama.cpp GitHub**: https://github.com/ggerganov/llama.cpp
- **Build Instructions**: https://github.com/ggerganov/llama.cpp#build
- **Examples**: https://github.com/ggerganov/llama.cpp/tree/master/examples

### Model Repositories

- **TheBloke (quantized models)**: https://huggingface.co/TheBloke
- **HuggingFace Hub**: https://huggingface.co/models?library=gguf
- **Official Meta models**: https://huggingface.co/meta-llama

### Community Resources

- **Reddit**: r/LocalLLaMA
- **Discord**: https://discord.gg/llama-cpp (check GitHub for invite)
- **Issues/Support**: https://github.com/ggerganov/llama.cpp/issues

### Video Tutorials

Search YouTube for:
- "llama.cpp tutorial"
- "build llama.cpp windows"
- "local LLM setup"

### Related Tools

- **Text Generation WebUI**: https://github.com/oobabooga/text-generation-webui
- **LM Studio**: https://lmstudio.ai/ (GUI for llama.cpp)
- **GPT4All**: https://gpt4all.io/ (Another GUI option)
- **Ollama**: https://ollama.ai/ (macOS/Linux focused)

### Learning Resources

- **Prompt Engineering Guide**: https://www.promptingguide.ai/
- **LLM Visualization**: https://bbycroft.net/llm
- **Understanding Quantization**: https://huggingface.co/docs/optimum/concept_guides/quantization

---

## Conclusion

Congratulations! You now have a comprehensive understanding of building and using llama.cpp on Windows.

**Quick recap:**

1. ✅ Install Visual Studio 2022 with C++ tools
2. ✅ Choose your build type (CPU/CUDA/Vulkan/OpenBLAS)
3. ✅ Use x64 Native Tools Command Prompt
4. ✅ Clone repository: `git clone https://github.com/ggerganov/llama.cpp.git`
5. ✅ Configure: `cmake .. -G "Visual Studio 17 2022" -A x64 [OPTIONS]`
6. ✅ Build: `cmake --build . --config Release`
7. ✅ Download models from HuggingFace
8. ✅ Run: `llama-cli.exe -m model.gguf -p "prompt" -ngl 99`

**Next steps:**

- Experiment with different models
- Try server mode for API access
- Integrate with your applications
- Join the community for support
- Star the repo and contribute back!

**Remember:**
- Start with small models for testing
- Monitor your VRAM/RAM usage
- Keep llama.cpp updated for latest features
- Read model licenses before commercial use
- Have fun with local LLMs!

---

## Appendix: Command Reference

### llama-cli.exe Common Parameters

```
-m, --model <FILE>          Model path (REQUIRED)
-p, --prompt <TEXT>         Prompt to generate from
-f, --file <FILE>           Load prompt from file
-n, --n-predict <N>         Tokens to generate (-1 = infinity, default: 128)
-c, --ctx-size <N>          Context size (default: 512)
-ngl, --n-gpu-layers <N>    Layers to offload to GPU (default: 0)
-t, --threads <N>           Thread count (default: auto)
-b, --batch-size <N>        Batch size for prompt processing (default: 512)
--temp <N>                  Temperature (default: 0.80)
--top-k <N>                 Top-K sampling (default: 40)
--top-p <N>                 Top-P sampling (default: 0.95)
--repeat-penalty <N>        Repetition penalty (default: 1.10)
--color                     Colorize output
--interactive               Run in interactive mode
--interactive-first         Interactive mode without prompt
-r, --reverse-prompt <TEXT> Stop generation at this text
--mlock                     Lock model in RAM
--no-mmap                   Don't use memory-mapped files
--verbose                   Verbose output
--help                      Show help
```

### llama-server.exe Common Parameters

```
-m, --model <FILE>          Model path (REQUIRED)
--host <ADDRESS>            Host to bind (default: 127.0.0.1)
--port <PORT>               Port to bind (default: 8080)
-c, --ctx-size <N>          Context size (default: 512)
-ngl, --n-gpu-layers <N>    Layers to offload to GPU
-t, --threads <N>           Thread count
-b, --batch-size <N>        Batch size
--embedding                 Enable embeddings endpoint
--log-disable               Disable request logging
--timeout <N>               Request timeout in seconds
```

### Environment Variables

```
CUDA_VISIBLE_DEVICES        Select GPU(s): "0" or "0,1"
GGML_CUDA_ENABLE_GRAPH      Enable CUDA graphs: "1"
GGML_VULKAN_DEVICE          Vulkan device: "0", "1", etc.
GGML_VULKAN_DEBUG           Enable Vulkan debug: "1"
OMP_NUM_THREADS             OpenMP threads
LLAMA_DEBUG                 Enable debug output: "1"
```

---

## Version Information

This guide was created for:
- **llama.cpp**: Latest (as of guide creation)
- **Visual Studio**: 2022 (Version 17.x)
- **Windows**: 10/11 (64-bit)
- **CMake**: 3.15+
- **CUDA**: 11.0+ (optional)
- **Vulkan**: 1.2+ (optional)

**Last updated**: February 2025

**Guide version**: 1.0

---

**Need help?** Feel free to ask questions or open issues on the llama.cpp GitHub repository!

**Found this helpful?** Star the repository and share with others!

Happy LLM-ing! 🦙🚀
