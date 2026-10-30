$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
function Find-Python([string]$name) {
  $a = Join-Path $root "runtime\$name\python.exe"
  $b = Join-Path $root "runtime\$name\Scripts\python.exe"
  if (Test-Path $a) { return $a }
  if (Test-Path $b) { return $b }
  throw "Missing $name runtime. Run Install_WhisperBox.bat first."
}
$cp = Find-Python 'chatterbox'
$wp = Find-Python 'whisper'
$env:HF_HUB_OFFLINE = '0'
$env:TRANSFORMERS_OFFLINE = '0'
$env:HF_HOME = Join-Path $root 'models\cache\huggingface'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
# Conda local-prefix DLL paths needed by native packages on Windows.
$env:PATH = (Join-Path $root 'runtime\chatterbox\Library\bin') + ';' + (Join-Path $root 'runtime\whisper\Library\bin') + ';' + $env:PATH
Write-Host 'Downloading WhisperBox model files. Internet access is required.' -ForegroundColor Cyan
& $cp (Join-Path $root 'app\download_models.py') --component chatterbox
if ($LASTEXITCODE -ne 0) { throw 'Chatterbox download failed. Retry Download_Models.bat; existing files are retained.' }
& $wp (Join-Path $root 'app\download_models.py') --component whisper
if ($LASTEXITCODE -ne 0) { throw 'Whisper download failed. Retry Download_Models.bat; existing files are retained.' }
Write-Host 'All required models are stored locally. Normal inference is offline.' -ForegroundColor Green
