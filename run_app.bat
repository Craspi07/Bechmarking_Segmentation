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

if not exist ".venv\Scripts\activate.bat" (
    echo [SETUP] Creating virtual environment in .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create the virtual environment.
        pause
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"

echo [SETUP] Installing/updating dependencies from requirements.txt ...
python -m pip install --upgrade pip >nul
pip install -r requirements.txt
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
    streamlit run app.py
    goto end
)
if "%choice%"=="2" (
    echo [RUN] Launching Advanced Benchmark app ...
    streamlit run app_advanced.py
    goto end
)
if "%choice%"=="3" (
    goto end
)

echo Invalid choice, please enter 1, 2, or 3.
goto menu

:end
endlocal
