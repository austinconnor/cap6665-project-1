@echo off
setlocal

cd /d "%~dp0"

if not defined UV_CACHE_DIR set "UV_CACHE_DIR=%CD%\.uv-cache"
if not defined PADDLE_PDX_CACHE_HOME set "PADDLE_PDX_CACHE_HOME=%CD%\outputs\model_cache\paddlex"
if not defined HF_HOME set "HF_HOME=%CD%\outputs\model_cache\huggingface"

where uv >nul 2>nul
if errorlevel 1 (
    echo uv not found; installing uv...
    powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%"
)

where uv >nul 2>nul
if errorlevel 1 (
    echo uv install finished, but uv.exe is not on PATH. Open a new terminal or add %%USERPROFILE%%\.local\bin to PATH.
    exit /b 1
)

uv sync
if errorlevel 1 exit /b %errorlevel%

if "%DOCMD_SETUP_ONLY%"=="1" (
    echo Setup complete.
    exit /b 0
)

uv run python scripts\run_demo.py %*
exit /b %errorlevel%