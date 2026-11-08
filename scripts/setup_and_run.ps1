﻿# setup_and_run.ps1
# Automates Windows environment setup and launches auto-mosaic GUI / CLI.

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AppArgs
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$rootDir = (Get-Item $scriptDir).Parent.FullName

Set-Location $rootDir

# Ensure Python operates in UTF-8 mode regardless of system locale/code page
$env:PYTHONUTF8 = "1"

try {
    Write-Host "==========================================================" -ForegroundColor Cyan
    Write-Host "       auto-mosaic: Windows Auto-Setup & Launcher         " -ForegroundColor Cyan
    Write-Host "==========================================================" -ForegroundColor Cyan

    # 1. Check or Setup Python & venv
    $venvPython = Join-Path $rootDir "venv\Scripts\python.exe"
    $venvPythonw = Join-Path $rootDir "venv\Scripts\pythonw.exe"
    $venvPip = Join-Path $rootDir "venv\Scripts\pip.exe"

    $needVenvSetup = $false
    if (-not (Test-Path $venvPython)) {
    $needVenvSetup = $true
} else {
    # Check if core packages are importable
    try {
        & $venvPython -c "import cv2, PySide6, onnxruntime" 2>$null
        if ($LASTEXITCODE -ne 0) {
            $needVenvSetup = $true
        }
    } catch {
        $needVenvSetup = $true
    }
}

if ($needVenvSetup) {
    Write-Host "`n[1/4] Setting up Python virtual environment..." -ForegroundColor Yellow

    # Check for system Python (>= 3.10)
    $systemPython = $null
    try {
        $pyCmd = Get-Command python -ErrorAction SilentlyContinue
        if ($pyCmd) {
            $pyVer = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
            $major, $minor = $pyVer.Split('.')
            if ([int]$major -ge 3 -and [int]$minor -ge 10) {
                $systemPython = "python"
            }
        }
    } catch {}

    if ($systemPython) {
        Write-Host "Found suitable system Python: $(& python --version)" -ForegroundColor Green
        Write-Host "Creating venv..."
        & python -m venv "$rootDir\venv"
    } else {
        # Setup portable Python embedded
        $embedDir = Join-Path $rootDir "python_embed"
        $embedPython = Join-Path $embedDir "python.exe"

        if (-not (Test-Path $embedPython)) {
            Write-Host "System Python not found. Setting up portable Python 3.11 embedded..." -ForegroundColor Yellow
            if (-not (Test-Path $embedDir)) { New-Item -ItemType Directory -Path $embedDir | Out-Null }

            $zipUrl = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip"
            $zipPath = Join-Path $rootDir "python-embed.zip"
            Write-Host "Downloading Python embedded package..."
            Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath

            Write-Host "Extracting Python embedded..."
            Expand-Archive -Path $zipPath -DestinationPath $embedDir -Force
            Remove-Item $zipPath -Force

            # Enable 'import site' in ._pth file so pip packages are recognized
            $pthFile = Get-ChildItem -Path $embedDir -Filter "*._pth" | Select-Object -First 1
            if ($pthFile) {
                $content = Get-Content $pthFile.FullName
                $newContent = @()
                foreach ($line in $content) {
                    if ($line -match "^#\s*import\s+site") {
                        $newContent += "import site"
                    } else {
                        $newContent += $line
                    }
                }
                # Ensure Lib/site-packages and root are included
                if ($newContent -notcontains "Lib\site-packages") { $newContent += "Lib\site-packages" }
                if ($newContent -notcontains ".") { $newContent += "." }
                $newContent | Set-Content $pthFile.FullName
            }

            # Install pip into embedded Python
            $getPipPath = Join-Path $embedDir "get-pip.py"
            Write-Host "Downloading get-pip.py..."
            Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $getPipPath
            Write-Host "Installing pip into portable environment..."
            & $embedPython $getPipPath --no-warn-script-location
            Remove-Item $getPipPath -Force

            # Install virtualenv into embedded Python to generate clean venvs
            Write-Host "Installing virtualenv..."
            & $embedPython -m pip install virtualenv --no-warn-script-location
        }

        Write-Host "Creating venv using portable Python..."
        & $embedPython -m virtualenv "$rootDir\venv"
    }

    Write-Host "Installing project requirements (PySide6, ONNX Runtime, OpenCV, etc.)..." -ForegroundColor Yellow
    & $venvPython -m pip install --upgrade pip
    $reqWinPath = if (Test-Path "$rootDir\requirements\requirements-windows.txt") { "$rootDir\requirements\requirements-windows.txt" } else { "$rootDir\requirements-windows.txt" }
    & $venvPython -m pip install -r $reqWinPath
    Write-Host "Python environment setup complete!" -ForegroundColor Green
} else {
    Write-Host "[1/4] Python environment is ready." -ForegroundColor Green
}

# 2. Check or Setup FFmpeg
Write-Host "`n[2/4] Checking FFmpeg / FFprobe..." -ForegroundColor Yellow
$binDir = Join-Path $rootDir "bin"
$ffmpegExe = Join-Path $binDir "ffmpeg.exe"
$ffprobeExe = Join-Path $binDir "ffprobe.exe"

$hasFfmpeg = (Test-Path $ffmpegExe) -or (Get-Command ffmpeg -ErrorAction SilentlyContinue)
$hasFfprobe = (Test-Path $ffprobeExe) -or (Get-Command ffprobe -ErrorAction SilentlyContinue)

if (-not ($hasFfmpeg -and $hasFfprobe)) {
    Write-Host "FFmpeg not detected. Downloading portable FFmpeg essentials..." -ForegroundColor Yellow
    if (-not (Test-Path $binDir)) { New-Item -ItemType Directory -Path $binDir | Out-Null }

    $ffmpegZip = Join-Path $env:TEMP "ffmpeg_setup.zip"
    $ffmpegExtract = Join-Path $env:TEMP "ffmpeg_setup_extracted"
    
    Invoke-WebRequest -Uri "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip" -OutFile $ffmpegZip
    Write-Host "Extracting FFmpeg..."
    Expand-Archive -Path $ffmpegZip -DestinationPath $ffmpegExtract -Force

    $foundFfmpeg = Get-ChildItem -Path $ffmpegExtract -Filter "ffmpeg.exe" -Recurse | Select-Object -First 1
    if ($foundFfmpeg) {
        Copy-Item "$($foundFfmpeg.DirectoryName)\*" -Destination $binDir -Force
    }
    Remove-Item $ffmpegZip -Force -ErrorAction SilentlyContinue
    Remove-Item $ffmpegExtract -Recurse -Force -ErrorAction SilentlyContinue

    Write-Host "FFmpeg installed to $binDir" -ForegroundColor Green
} else {
    Write-Host "FFmpeg is ready." -ForegroundColor Green
}

# Ensure bin directory is in current session PATH
if (Test-Path $binDir) {
    if ($env:PATH -notlike "*$binDir*") {
        $env:PATH = "$binDir;" + $env:PATH
    }
}

# 3. Check or Download SCRFD & License Plate Models
Write-Host "`n[3/4] Checking AI Detection Models (Face & License Plate)..." -ForegroundColor Yellow
$modelsDir = Join-Path $rootDir "models"
$defaultModel = Join-Path $modelsDir "scrfd_10g_bnkps.onnx"
$plateModel = Join-Path $modelsDir "license-plate-finetune-v1n.onnx"

if (-not (Test-Path $defaultModel) -or -not (Test-Path $plateModel)) {
    Write-Host "Downloading AI detection models (this only happens on first run)..." -ForegroundColor Yellow
    & $venvPython "$rootDir\scripts\download_models.py"
    Write-Host "Models successfully downloaded!" -ForegroundColor Green
} else {
    Write-Host "Face & License Plate detection models are ready." -ForegroundColor Green
}

# 4. Launch Application
Write-Host "`n[4/4] Launching auto-mosaic..." -ForegroundColor Cyan
Write-Host "==========================================================`n" -ForegroundColor Cyan

if ($AppArgs -and $AppArgs.Count -gt 0) {
    & $venvPython -m face_mosaic.cli @AppArgs
} else {
    if (-not (Test-Path $venvPythonw)) {
        $venvPythonw = $venvPython
    }
    # Pre-flight check: ensure GUI module can be imported without fatal errors
    $checkOutput = & $venvPython -c "import face_mosaic.gui.app" 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "アプリケーションの読み込みに失敗しました:`n$checkOutput"
    }
    # Start GUI detached with pythonw (no console window)
    Start-Process -FilePath $venvPythonw -ArgumentList @("-m", "face_mosaic.gui.app") -WorkingDirectory $rootDir
}
} catch {
    $errMsg = $_.Exception.Message
    try {
        Add-Type -AssemblyName System.Windows.Forms
        [System.Windows.Forms.MessageBox]::Show(
            "起動時または処理中にエラーが発生しました:`n`n$errMsg",
            "auto-mosaic エラー",
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Error
        ) | Out-Null
    } catch {
        Write-Error "Launcher Error: $errMsg"
    }
    exit 1
}
