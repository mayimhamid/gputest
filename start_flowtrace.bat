@echo off
rem FlowTrace one-click launcher for Windows.
rem  - creates a private Python environment in .venv (nothing is installed system-wide)
rem  - prefers a Python version with prebuilt CuPy/Numba wheels (3.13 .. 3.10)
rem  - installs FlowTrace with the CPU accelerators, AMG, HDF5 and report extras
rem  - if an NVIDIA GPU is present (nvidia-smi), also installs the CUDA backend
rem  - reinstalls only when the FlowTrace version or the GPU situation changes
rem Set FLOWTRACE_LAUNCHER_DRYRUN=1 to install/check without starting the dashboard.

setlocal EnableDelayedExpansion
cd /d "%~dp0"
set "VENV=%~dp0.venv"
set "PYV=%VENV%\Scripts\python.exe"

if exist "%PYV%" goto have_venv
set "PY="
for %%V in (3.13 3.12 3.11 3.10) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>nul && set "PY=py -%%V"
  )
)
if not defined PY (
  where python >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo Python was not found. Install Python 3.10-3.13 from https://www.python.org and tick "Add to PATH".
  pause
  exit /b 1
)
echo Creating a private environment in .venv with !PY! ...
!PY! -m venv "%VENV%"
if errorlevel 1 goto fail

:have_venv
for /f "tokens=3" %%i in ('findstr /b /c:"__version__" flowtrace\__init__.py') do set "FTVER=%%~i"
set "WANT=%FTVER%-cpu"
where nvidia-smi >nul 2>nul && set "WANT=%FTVER%-gpu"
set "STAMP=%VENV%\.flowtrace_installed"
set "HAVE="
if exist "%STAMP%" set /p HAVE=<"%STAMP%"
if not "!HAVE!"=="!WANT!" (
  echo Installing FlowTrace !FTVER! and dependencies ...
  "%PYV%" -m pip install --upgrade pip
  "%PYV%" -m pip install -e ".[cpu,amg,h5,report]"
  if errorlevel 1 goto fail
  if "!WANT:~-4!"=="-gpu" (
    echo NVIDIA GPU detected: installing the CUDA backend ^(CuPy for CUDA 12^) ...
    "%PYV%" -m pip install "cupy-cuda12x[ctk]" nvidia-ml-py
    if errorlevel 1 echo GPU backend could not be installed - FlowTrace will run on the CPU.
  )
  >"%STAMP%" echo !WANT!
)
if "%FLOWTRACE_LAUNCHER_DRYRUN%"=="1" (
  "%PYV%" -m flowtrace --version
  exit /b 0
)
"%PYV%" -m flowtrace ui %*
pause
exit /b 0

:fail
echo Installation failed. See the messages above; you can also run:  pip install -e ".[cpu,amg]"
pause
exit /b 1
