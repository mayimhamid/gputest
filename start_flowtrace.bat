@echo off
rem FlowTrace one-click launcher for Windows.
rem First run: installs FlowTrace and the recommended CPU accelerators.
rem GPU: install the CuPy wheel that matches your CUDA driver, e.g.  pip install cupy-cuda12x

setlocal
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.9+ from https://www.python.org and tick "Add to PATH".
  pause
  exit /b 1
)
if not exist ".flowtrace_installed" (
  echo Installing FlowTrace and dependencies ...
  python -m pip install --upgrade pip
  python -m pip install -e . numba psutil
  if errorlevel 1 (
    echo Installation failed.
    pause
    exit /b 1
  )
  echo ok> .flowtrace_installed
)
python -m flowtrace ui %*
pause
