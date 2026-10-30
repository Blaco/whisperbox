$ErrorActionPreference = 'Stop'
# Cache and target environments may be on different drives. Copying is intentional.
$env:UV_LINK_MODE = 'copy'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
Write-Host '=== WhisperBox private runtime installer ===' -ForegroundColor Cyan
Write-Host 'All WhisperBox environments and models are installed inside this folder.'
Write-Host 'Automatic setup can download private Python 3.10 and 3.11 runtimes.'
Write-Host ''
# Capture errors on installations launched by double-click; logs stay local.
New-Item -ItemType Directory -Force -Path (Join-Path $root 'logs') | Out-Null
$installLog = Join-Path $root 'logs\install_transcript.log'
try { Start-Transcript -Path $installLog -Append -ErrorAction Stop | Out-Null } catch { Write-Warning $_ }
$sys = Get-CimInstance Win32_OperatingSystem
if ([Environment]::Is64BitOperatingSystem -ne $true) { throw '64-bit Windows is required.' }
if ([int]$sys.BuildNumber -lt 17763) { throw 'Windows 10 build 1809 or newer is required.' }
$drive = ([IO.Path]::GetPathRoot($root))
$free = (Get-PSDrive -Name $drive.Substring(0,1)).Free
if ($free -lt 15GB) { Write-Warning 'Less than 15 GB free disk space. Installation may fail during dependency/model downloads.' }
Write-Host 'Environment setup:' -ForegroundColor Cyan
Write-Host ' [1] Automatic private Python (recommended; downloads Python if needed)'
Write-Host ' [2] Existing Conda (uses your installed Conda)'
Write-Host ' [3] Existing Python / venv (requires Python 3.10 and 3.11)'
$backendChoice = Read-Host 'Select environment method [1]'
switch ($backendChoice) {
    '2' { $backend = 'conda' }
    '3' { $backend = 'existing-venv' }
    default { $backend = 'private-venv' }
}
$device = Read-Host 'Inference: [1] CPU (works without NVIDIA, default) or [2] NVIDIA CUDA'
if ($device -eq '2') { $device = 'cuda' } else { $device = 'cpu' }
Write-Host "Starting $backend installation for $device in $root ..." -ForegroundColor Cyan

function Run($exe, [string[]]$arguments) {
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $exe $($arguments -join ' ')" }
}
function Py($name, [string[]]$arguments) {
    $python = Join-Path $root "runtime\$name\Scripts\python.exe"
    if (-not (Test-Path $python)) { throw "Python missing at $python" }
    Run $python $arguments
}
New-Item -ItemType Directory -Force -Path 'runtime','models\voices','models\cache','config','data\recordings','data\transcripts','data\generated_audio','logs' | Out-Null

if ($backend -in @('private-venv','existing-venv')) {
    # uv provides managed, portable Python runtimes without launching a system installer.
    # All interpreters, virtual environments, and uv itself remain under runtime/.
    if ($backend -eq 'private-venv') {
        $uvToolsDir = Join-Path $root 'runtime\_tools'
        $privateUv = Join-Path $uvToolsDir 'uv.exe'
        $env:UV_PYTHON_INSTALL_DIR = Join-Path $root 'runtime\_managed_python'
        # Keep uv's managed-Python executable links inside WhisperBox too.
        $env:UV_PYTHON_BIN_DIR = Join-Path $root 'runtime\_tools'
        New-Item -ItemType Directory -Force -Path $uvToolsDir,$env:UV_PYTHON_INSTALL_DIR | Out-Null
        if (-not (Test-Path $privateUv)) {
            $uvArchive = Join-Path $uvToolsDir 'uv-windows-x64.zip'
            $uvUnpack = Join-Path $uvToolsDir '_uv_unpack'
            $uvUrl = 'https://github.com/astral-sh/uv/releases/download/0.9.18/uv-x86_64-pc-windows-msvc.zip'
            Write-Host 'Downloading uv bootstrap for private Python installation ...'
            try {
                Invoke-WebRequest -Uri $uvUrl -OutFile $uvArchive -UseBasicParsing -ErrorAction Stop
                if ((Get-Item $uvArchive).Length -lt 1000000) { throw 'uv bootstrap download was incomplete' }
                if (Test-Path $uvUnpack) { Remove-Item $uvUnpack -Recurse -Force }
                Expand-Archive -LiteralPath $uvArchive -DestinationPath $uvUnpack -Force
                $uvExtracted = Get-ChildItem -Path $uvUnpack -Filter uv.exe -File -Recurse | Select-Object -First 1
                if (-not $uvExtracted) { throw 'uv.exe was not found in downloaded archive' }
                Copy-Item -LiteralPath $uvExtracted.FullName -Destination $privateUv -Force
            } catch {
                throw "Unable to download uv for automatic Python setup: $_. Check network access to github.com, or choose existing Conda / Python."
            } finally {
                Remove-Item -LiteralPath $uvArchive -Force -ErrorAction SilentlyContinue
                Remove-Item -LiteralPath $uvUnpack -Recurse -Force -ErrorAction SilentlyContinue
            }
        }
        Run $privateUv @('--version')
        Write-Host 'Installing private Python 3.10 and 3.11 using uv ...'
        Run $privateUv @('python','install','3.10','3.11')
    }
    # Option 3 uses existing Python, including pyenv-win shims and uv-managed Python.
    # Resolve every candidate to a REAL python.exe before passing it to venv.
    function Test-PythonCandidate([string]$candidate, [string]$version, [string[]]$prefixArgs = @()) {
        if (-not $candidate) { return $null }
        $code = 'import sys,os; assert sys.version_info[:2] == (' + ($version -replace '\.', ',') + ') and sys.maxsize > 2**32; print(os.path.realpath(sys.executable))'
        try {
            $result = @(& $candidate @prefixArgs '-c' $code 2>$null)
            if ($LASTEXITCODE -ne 0 -or $result.Count -eq 0) { return $null }
            $actual = [string]($result | Select-Object -Last 1)
            $actual = $actual.Trim()
            if (-not (Test-Path -LiteralPath $actual -PathType Leaf)) { return $null }
            if (-not $actual.EndsWith('python.exe', [StringComparison]::OrdinalIgnoreCase)) { return $null }
            return $actual
        } catch { return $null }
    }
    function Find-BasePython([string]$version) {
        $seen = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
        $candidates = New-Object 'System.Collections.Generic.List[string]'
        function Add-PythonCandidate([string]$candidate) {
            if ($candidate -and $seen.Add($candidate)) { $candidates.Add($candidate) }
        }

        # pyenv-win installs real CPython executables in versions\<version>\python.exe.
        # Find these directly, rather than relying on .bat shims for virtualenv creation.
        $pyenvRoots = @($env:PYENV_ROOT, (Join-Path $HOME '.pyenv\pyenv-win'),
                        (Join-Path $HOME '.pyenv\pyenv-win')) | Where-Object { $_ }
        foreach ($pyenvRoot in $pyenvRoots) {
            $versionsDir = Join-Path $pyenvRoot 'versions'
            if (Test-Path $versionsDir) {
                Get-ChildItem -LiteralPath $versionsDir -Directory -ErrorAction SilentlyContinue |
                    Where-Object { $_.Name -match ('^' + [regex]::Escape($version) + '(\.|$)') } |
                    Sort-Object Name -Descending |
                    ForEach-Object { Add-PythonCandidate (Join-Path $_.FullName 'python.exe') }
            }
        }

        # Standard per-user/all-users installs and uv's managed interpreter layout.
        $minor = $version.Replace('.', '')
        foreach ($base in @($env:LOCALAPPDATA, $env:ProgramFiles, ${env:ProgramFiles(x86)})) {
            if (-not $base) { continue }
            Add-PythonCandidate (Join-Path $base "Programs\Python\Python$minor\python.exe")
            Add-PythonCandidate (Join-Path $base "Python$minor\python.exe")
        }
        $uvRoots = @($env:UV_PYTHON_INSTALL_DIR, (Join-Path $HOME 'AppData\Roaming\uv\python'),
                     (Join-Path $HOME 'AppData\Local\uv\python'),
                     (Join-Path $HOME '.local\share\uv\python'),
                     (Join-Path $root 'runtime\_managed_python')) | Where-Object { $_ }
        foreach ($uvRoot in $uvRoots) {
            if (-not (Test-Path $uvRoot)) { continue }
            Get-ChildItem -LiteralPath $uvRoot -Directory -Filter "cpython-$version.*" -ErrorAction SilentlyContinue |
                ForEach-Object { Add-PythonCandidate (Join-Path $_.FullName 'python.exe'); Add-PythonCandidate (Join-Path $_.FullName 'install\python.exe') }
        }

        # Check launcher and named commands (including pyenv-win's .bat shims).
        $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
        if ($launcher) {
            $actual = Test-PythonCandidate $launcher.Source $version @("-$version")
            if ($actual) { Add-PythonCandidate $actual }
        }
        foreach ($name in @("python$version", "python$minor", 'python')) {
            foreach ($cmd in @(Get-Command $name -All -ErrorAction SilentlyContinue)) {
                if ($cmd.CommandType -ne 'Application') { continue }
                $actual = Test-PythonCandidate $cmd.Source $version
                if ($actual) { Add-PythonCandidate $actual }
            }
        }
        foreach ($candidate in $candidates) {
            $actual = Test-PythonCandidate $candidate $version
            if ($actual) { return $actual }
        }
        return $null
    }
    $basePython = @{}
    if ($backend -eq 'existing-venv') {
        foreach ($v in @('3.10','3.11')) {
            $found = Find-BasePython $v
            if (-not $found) {
                Write-Warning "Unable to auto-detect 64-bit Python $v. Enter the path to its python.exe (leave blank to cancel)."
                $manual = Read-Host "Path to Python $v python.exe"
                $found = Test-PythonCandidate $manual.Trim(' ', '"', "'") $v
                if (-not $found) {
                    throw "A valid 64-bit Python $v executable was not provided. Option 3 does not download Python."
                }
            }
            $basePython[$v] = $found
            Write-Host "Python $v verified: $found" -ForegroundColor Green
        }
    }
    # A Scripts\python.exe file alone does not mean a venv is usable.
    # A previous Conda install, an interrupted venv creation, or a copied runtime
    # can leave that executable present without the critical pyvenv.cfg file.
    function Test-ExistingVenv([string]$envPath, [string]$version) {
        $python = Join-Path $envPath 'Scripts\python.exe'
        $cfg = Join-Path $envPath 'pyvenv.cfg'
        if (-not (Test-Path -LiteralPath $cfg -PathType Leaf) -or
            -not (Test-Path -LiteralPath $python -PathType Leaf)) { return $false }
        $check = 'import sys,venv; assert sys.version_info[:2] == (' + ($version -replace '\.', ',') + '); assert sys.prefix != sys.base_prefix'
        try {
            & $python '-c' $check 2>$null | Out-Null
            return ($LASTEXITCODE -eq 0)
        } catch { return $false }
    }
    foreach ($spec in @(@('gui','3.11'), @('whisper','3.11'), @('chatterbox','3.10'))) {
        $envname = $spec[0]; $version = $spec[1]
        $envPath = Join-Path $root "runtime\$envname"
        if (Test-ExistingVenv $envPath $version) {
            Write-Host "Reusing verified $envname virtual environment." -ForegroundColor Green
            continue
        }
        if (Test-Path -LiteralPath $envPath) {
            $suffix = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
            $backup = Join-Path $root "runtime\_previous_${envname}_$suffix"
            Write-Warning "Existing $envname environment is incomplete, invalid, or not a venv for Python $version."
            Write-Host "Moving it aside to $backup (nothing deleted)." -ForegroundColor Yellow
            Move-Item -LiteralPath $envPath -Destination $backup -ErrorAction Stop
        }
        Write-Host "Creating $envname with Python $version ..."
        if ($backend -eq 'private-venv') {
            Run $privateUv @('venv','--python',$version,'--managed-python','--seed',"runtime\$envname")
        } else {
            # Always use the resolved interpreter rather than a pyenv-win .bat shim.
            Run $basePython[$version] @('-m','venv',"runtime\$envname")
        }
        if (-not (Test-ExistingVenv $envPath $version)) {
            throw "The new $envname virtual environment did not pass Python $version validation."
        }
    }
} else {
    if (-not (Get-Command conda -ErrorAction SilentlyContinue)) { throw 'Conda is not on PATH. Use Anaconda Prompt or choose automatic private Python.' }
    foreach ($spec in @(@('gui','3.11'), @('whisper','3.11'), @('chatterbox','3.10'))) {
        $envname = $spec[0]; $version = $spec[1]
        $envPath = Join-Path $root "runtime\$envname"
        if (-not (Test-Path (Join-Path $envPath 'python.exe')) -and -not (Test-Path (Join-Path $envPath 'Scripts\python.exe'))) {
            Run 'conda' @('create', '-y', '-p', $envPath, "python=$version", 'pip')
        }
    }
}

# Both venv and Conda layouts resolve to the same three environment names.
$resolvedPython = @{}
$expectedVersions = @{ gui='3.11'; whisper='3.11'; chatterbox='3.10' }
foreach ($envname in @('gui','whisper','chatterbox')) {
    $candidates = @((Join-Path $root "runtime\$envname\Scripts\python.exe"),(Join-Path $root "runtime\$envname\python.exe"))
    $resolvedPython[$envname] = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $resolvedPython[$envname]) { throw "No Python for $envname" }
    $version = $expectedVersions[$envname]
    Run $resolvedPython[$envname] @('-c',("import sys; assert sys.version_info[:2] == (" + ($version -replace '\.', ',') + "), 'Wrong Python minor version for $envname'"))
}

# Bootstrap uv once in the GUI Python; uv manages all three target environments.
# We retain pip only for bootstrapping uv, not for installing application dependencies.
Write-Host 'Bootstrapping uv package manager ...'
Run $resolvedPython['gui'] @('-m','pip','install','--upgrade','uv==0.9.18')
function Uv([string]$envname, [string[]]$requirements) {
    $target = $resolvedPython[$envname]
    $uvArgs = @('-m','uv','pip','install','--python',$target) + $requirements
    Run $resolvedPython['gui'] $uvArgs
}
Write-Host 'Installing GUI components with uv ...'
Uv 'gui' @('-r',(Join-Path $root 'app\requirements-gui.txt'))

Write-Host 'Verifying GUI WASAPI dependency ...'
Run $resolvedPython['gui'] @('-c', 'import soundcard')
Write-Host 'GUI SoundCard ready' -ForegroundColor Green

Write-Host 'Installing Whisper and Silero with uv ...'
if ($device -eq 'cpu') {
    Uv 'whisper' @('torch==2.6.0','--index-url','https://download.pytorch.org/whl/cpu')
} else {
    Uv 'whisper' @('torch==2.6.0','--index-url','https://download.pytorch.org/whl/cu126')
}
Uv 'whisper' @('faster-whisper==1.2.1','silero-vad','onnxruntime','numpy<2','sounddevice','soundfile','soundcard>=0.4,<1','librosa>=0.10.2,<0.12','pedalboard>=0.9.23,<1')

Write-Host 'Verifying Whisper WASAPI dependency ...'
Run $resolvedPython['whisper'] @('-c', 'import soundcard')
Write-Host 'Whisper SoundCard ready' -ForegroundColor Green

# Install the desired torch wheels before Chatterbox dependency resolution.
# uv preserves installed dependencies that already satisfy chatterbox-tts==0.1.7;
# CUDA local-version wheels (2.6.0+cu126) satisfy chatterbox's ==2.6.0 pins.
Write-Host "Installing $device PyTorch and torchaudio for Chatterbox with uv ..."
if ($device -eq 'cpu') {
    Uv 'chatterbox' @('torch==2.6.0','torchaudio==2.6.0','--index-url','https://download.pytorch.org/whl/cpu')
} else {
    Uv 'chatterbox' @('torch==2.6.0','torchaudio==2.6.0','--index-url','https://download.pytorch.org/whl/cu126')
}
Write-Host 'Installing pinned Chatterbox 0.1.7 with uv (reusing existing PyTorch) ...'
Uv 'chatterbox' @('chatterbox-tts==0.1.7')
# Do not silently accept a CPU torch build on a requested CUDA installation.
if ($device -eq 'cuda') {
    Run $resolvedPython['chatterbox'] @('-c', 'import torch,torchaudio; assert torch.version.cuda is not None, "Chatterbox torch CUDA build was replaced"; assert "+cu126" in torch.__version__, "Unexpected Chatterbox torch build"; assert "+cu126" in torchaudio.__version__, "Unexpected torchaudio build"')
} else {
    Run $resolvedPython['chatterbox'] @('-c', 'import torch,torchaudio; assert torch.version.cuda is None, "CPU mode unexpectedly contains CUDA torch"')
}
Write-Host 'Applying Chatterbox / Perth compatibility fix (setuptools 80.9.0) ...'
Uv 'chatterbox' @('setuptools==80.9.0')

# Verify Perth loads a real watermarker instead of silently setting it to None.
# Verify the required watermarking dependency.
Write-Host 'Verifying Perth watermarker import ...'
Run $resolvedPython['chatterbox'] @('-c', 'import pkg_resources; import perth; assert callable(perth.PerthImplicitWatermarker)')
Write-Host 'Perth watermarker verified' -ForegroundColor Green

# Fetch model weights once, while installer is explicitly online.
# During GUI inference local model paths are used and Hugging Face is forced offline.
Write-Host 'Downloading Original, Turbo and Whisper small.en models locally ...'
$env:HF_HUB_OFFLINE = '0'
$env:TRANSFORMERS_OFFLINE = '0'
$env:HF_HOME = Join-Path $root 'models\cache\huggingface'
Run $resolvedPython['chatterbox'] @((Join-Path $root 'app\download_models.py'),'--component','chatterbox')
Run $resolvedPython['whisper'] @((Join-Path $root 'app\download_models.py'),'--component','whisper')

# Verify imports in both worker environments, rather than just existence of executables.
Run $resolvedPython['whisper'] @('-c', 'import faster_whisper, silero_vad, torch, soundcard, soundfile')
Run $resolvedPython['chatterbox'] @('-c', 'import chatterbox, torch, torchaudio, perth, soundfile')
if ($device -eq 'cuda') {
    foreach ($envname in @('whisper','chatterbox')) {
        Run $resolvedPython[$envname] @('-c', 'import torch; assert torch.cuda.is_available(), "CUDA not available"; a=torch.ones(1,device="cuda"); assert a.item()==1')
    }
}
# Validate the actual WAV file, not merely its existence.
$defaultVoice = Join-Path $root 'models\voices\Chatterbox_Default.wav'
function Test-DefaultVoice {
    if (-not (Test-Path $defaultVoice)) { return $false }
    & $resolvedPython['chatterbox'] '-c' 'import soundfile as sf,sys; info=sf.info(sys.argv[1]); assert info.frames >= 16000 and info.samplerate >= 16000 and info.channels >= 1' $defaultVoice 2>$null
    return ($LASTEXITCODE -eq 0)
}
if (Test-DefaultVoice) {
    Write-Host "Default reference WAV already exists and is valid: $defaultVoice" -ForegroundColor Green
} else {
    if (Test-Path $defaultVoice) {
        $backup = "$defaultVoice.invalid.$(Get-Date -Format 'yyyyMMdd_HHmmss').bak"
        Move-Item -LiteralPath $defaultVoice -Destination $backup
        Write-Warning "Existing default WAV was invalid; backed it up to: $backup"
    }
    Write-Host 'Generating Chatterbox built-in voice sample (one-time; may take time on CPU) ...'
    try {
        Run $resolvedPython['chatterbox'] @((Join-Path $root 'app\create_default_voice.py'),'--device',$device)
        if (-not (Test-DefaultVoice)) { throw 'Generated WAV failed audio format/duration validation' }
        Write-Host "Created and validated Chatterbox default reference: $defaultVoice" -ForegroundColor Green
    } catch {
        Write-Host "FAILED to generate Chatterbox default reference: $_" -ForegroundColor Red
        Write-Host 'Re-run the installer to retry generating the default WAV.' -ForegroundColor Yellow
        throw
    }
}

# Write selected compute setting without discarding user customizations.
$configFile = Join-Path $root 'config\settings.json'
if (Test-Path $configFile) { $settings = Get-Content $configFile -Raw | ConvertFrom-Json } else { $settings = [pscustomobject]@{} }
$settings | Add-Member -NotePropertyName 'compute_device' -NotePropertyValue $device -Force
$existingVoice = $settings.PSObject.Properties['voice']
if (-not $existingVoice -or -not $existingVoice.Value) {
    $settings | Add-Member -NotePropertyName 'voice' -NotePropertyValue $defaultVoice -Force
}
[System.IO.File]::WriteAllText($configFile, ($settings | ConvertTo-Json -Depth 10), (New-Object System.Text.UTF8Encoding($false)))
$manifest = [pscustomobject]@{ method = $backend; device = $device; created = (Get-Date).ToString('o') }
[System.IO.File]::WriteAllText((Join-Path $root 'config\install.json'), ($manifest | ConvertTo-Json), (New-Object System.Text.UTF8Encoding($false)))
Write-Host ''
Write-Host 'Installation complete. Run Launch_WhisperBox.vbs.' -ForegroundColor Green
Write-Host 'If the folder is moved and Python runtimes fail, rerun Install_WhisperBox.bat to repair.'
try { Stop-Transcript | Out-Null } catch {}
