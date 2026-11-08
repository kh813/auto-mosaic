@echo off
setlocal
cd /d "%~dp0"

:: If command-line arguments are provided, run synchronously in CLI mode (in this console)
if "%~1" neq "" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_and_run.ps1" %*
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo [Error] Execution stopped with error code %ERRORLEVEL%.
        pause
    )
    exit /b %ERRORLEVEL%
)

:: If first run (venv does not exist or models missing), show console so the user sees setup progress
if not exist "%~dp0venv\Scripts\python.exe" goto first_run
if not exist "%~dp0models\scrfd_10g_bnkps.onnx" goto first_run
if not exist "%~dp0models\license-plate-finetune-v1n.onnx" goto first_run

:: Subsequent runs: launch GUI silently in background
start "" wscript.exe //nologo "%~dp0scripts\run_gui.vbs"
exit /b 0

:first_run
echo ========================================================
echo   auto-mosaic: Initializing setup on first run...
echo   Please wait while dependencies and models are set up.
echo ========================================================
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_and_run.ps1"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [Error] Setup failed.
    pause
    exit /b %ERRORLEVEL%
)
exit /b 0
