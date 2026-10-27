# Installation

FlowTrace needs Python 3.9 or newer, NumPy and SciPy. Everything else is optional.

```bash
pip install flowtrace                       # core: NumPy + SciPy
pip install "flowtrace[cpu]"                # + Numba JIT kernels and psutil (recommended)
pip install "flowtrace[amg]"                # + pyamg (faster AMG setup on the CPU)
pip install "flowtrace[h5]"                 # + h5py (Fluent .cas.h5 / .dat.h5)
pip install "flowtrace[report]"             # + matplotlib (plots in validation reports)
pip install "flowtrace[gpu]"                # + CuPy for CUDA 12 with headers (NVIDIA GPUs)
pip install "flowtrace[fluent]"             # + PyFluent (live link to a licensed Fluent)
pip install "flowtrace[all]"                # cpu + amg + h5 + report
```

From a source checkout, use `pip install -e ".[cpu,amg,h5,report,dev]"`.

## Windows: one click

Double-click `start_flowtrace.bat`. On the first run it:

1. creates a private environment in `.venv`, preferring Python 3.13 down to 3.10, because those
   versions have prebuilt CuPy and Numba wheels;
2. installs FlowTrace with the CPU, AMG, HDF5 and report extras;
3. installs the CUDA backend if `nvidia-smi` finds an NVIDIA GPU;
4. opens the dashboard.

Later starts only reinstall when the FlowTrace version or the GPU situation changes.

## Native kernels (optional, recommended)

With a C++ compiler on the `PATH` (g++ or clang++; on Windows, MSYS2's UCRT64 g++), FlowTrace
compiles its C++/OpenMP kernels for the unstructured solver the first time they are needed,
in a few seconds, and caches them in `~/.flowtrace/native/`. On an NVIDIA GPU the CUDA
kernels are compiled by NVRTC, with no compiler required. Without a compiler, the same solver
runs on NumPy/SciPy and gives the same results. See [Native kernels](user-guide/native.md).

```bash
pacman -S mingw-w64-ucrt-x86_64-gcc        # Windows, from an MSYS2 shell (then add C:\msys64\ucrt64\bin to PATH)
sudo apt install g++                       # Debian/Ubuntu
```

## GPU notes

* CuPy wheels exist for specific Python versions. On a very new Python (for example 3.15 at the
  time of writing) there may be no CuPy wheel yet, so use 3.13 or older.
* `cupy-cuda12x[ctk]` also installs the CUDA headers. NVRTC needs them to compile FlowTrace's
  fused CUDA kernels at run time; no compiler or CUDA toolkit installation is required.
* CUDA 12 runtimes run on CUDA 12.0+ drivers (minor-version compatibility). This was measured
  on an RTX 4060 Laptop GPU.
* Check what was detected with `flowtrace probe`, and measure with `flowtrace gpu-report`.

## Check the installation

```bash
flowtrace --version
flowtrace probe
pytest -q                          # from a source checkout: about 230 tests (GPU tests skip without a GPU)
flowtrace benchmark --tier quick   # the verification suite (a few minutes)
```
