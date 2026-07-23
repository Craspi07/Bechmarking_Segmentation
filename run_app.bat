@echo off
setlocal enabledelayedexpansion

REM ===========================================================================
REM Segmentation Benchmarking Suite - Windows launcher
REM
REM Creates (or reuses) a local virtual environment, installs/updates the
REM dependencies from requirements.txt, then lets you choose which Streamlit
REM app to launch:
REM   1. Basic Benchmark    (app.py)          - crop/count, morphology, SNR,
REM                                              synthetic dots, replicates
REM   2. Advanced Benchmark (app_advanced.py) - consensus/STAPLE, perturbation
REM                                              stability, quality scoring, RCA
REM ===========================================================================

cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found on PATH. Install Python 3.9+ from python.org
    echo         and make sure "Add python.exe to PATH" is checked during setup.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [SETUP] Creating virtual environment in .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create the virtual environment.
        pause
        exit /b 1
    )
)

set "VENV_PY=%~dp0.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo [ERROR] Virtual environment python.exe not found at "%VENV_PY%".
    echo         Delete the .venv folder and re-run this script.
    pause
    exit /b 1
)

REM NOTE: everything below is invoked as "python.exe -m <module>", never as
REM a bare "pip" / "streamlit" command. Those bare commands resolve through
REM launcher .exe stubs (pip.exe, streamlit.exe) that have an *absolute*
REM interpreter path baked in at install time. If that interpreter later
REM moves, gets uninstalled, or is a fragile bundled copy (a common example:
REM the Python shipped inside Visual Studio at
REM "C:\Program Files (x86)\Microsoft Visual Studio\Shared\PythonXX_64\"),
REM the stub fails with a cryptic "Fatal error in launcher: Unable to create
REM process" instead of a normal Python error. Calling "python.exe -m X"
REM runs the module directly inside the given interpreter and never touches
REM those stubs, so this works regardless of what else is on PATH.

echo [SETUP] Installing/updating dependencies from requirements.txt ...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip inside the virtual environment.
    pause
    exit /b 1
)

"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency installation failed. See the output above for details.
    pause
    exit /b 1
)

:menu
echo.
echo ============================================================
echo   Segmentation Benchmarking Suite
echo ============================================================
echo   1. Basic Benchmark
echo        Crop/count, morphology QC, SNR/contrast, synthetic
echo        dot simulation, replicate consistency
echo   2. Advanced Benchmark
echo        Consensus/STAPLE, perturbation stability, boundary
echo        and homogeneity quality score, RCA proxy
echo   3. Exit
echo ============================================================
set "choice="
set /p choice="Select an option [1-3]: "

if "%choice%"=="1" (
    echo [RUN] Launching Basic Benchmark app ...
    "%VENV_PY%" -m streamlit run app.py
    goto end
)
if "%choice%"=="2" (
    echo [RUN] Launching Advanced Benchmark app ...
    "%VENV_PY%" -m streamlit run app_advanced.py
    goto end
)
if "%choice%"=="3" (
    goto end
)

echo Invalid choice, please enter 1, 2, or 3.
goto menu

:end
endlocal
