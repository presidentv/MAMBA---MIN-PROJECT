# End-to-end run on the FULL DAiSEE release on Windows: fine-tuned ViT-B/16 +
# Mamba, T=32. Same steps, in the same order, as scripts/run_full_daisee.sh.
#
#   $env:DAISEE_ROOT = "D:\DAiSEE"          # contains DataSet\ and Labels\
#   powershell -ExecutionPolicy Bypass -File scripts\run_full_daisee.ps1
#
# Safe to re-run after an interruption (power cut, reboot, Ctrl+C). Stage 1
# skips clips already cached, and training resumes from
# checkpoints\<run>\last.pt. For a completely fresh run, delete
# checkpoints\<run> first.
#
# Optional environment:
#   $env:RUN = "full_ft32"      run name (checkpoints\, logs\, artifacts\ file names)
#   $env:STAGE1_SHARDS = "6"    parallel Stage 1 processes (default: cores - 1, max 8)
#   $env:WORKERS = "4"          DataLoader workers during training/evaluation
#   $env:PYTHON = "..."         interpreter (default: .venv\Scripts\python.exe if present)
#   $env:MRG_CACHE_DIR / $env:MRG_CHECKPOINT_DIR   see configs\config_full.yaml
#
# Written for Windows PowerShell 5.1, which ships with Windows 10/11.

$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

$Config = "configs/config_full.yaml"
$Run = if ($env:RUN) { $env:RUN } else { "full_ft32" }
$Workers = if ($env:WORKERS) { $env:WORKERS } else { "4" }

# Leave one core for the OS; more than 8 MediaPipe processes gains little and
# each holds its own model in RAM.
$Cores = [int]$env:NUMBER_OF_PROCESSORS
$DefaultShards = [Math]::Max(1, [Math]::Min(8, $Cores - 1))
$Shards = if ($env:STAGE1_SHARDS) { [int]$env:STAGE1_SHARDS } else { $DefaultShards }

if ($env:PYTHON) {
    $Py = $env:PYTHON
} elseif (Test-Path ".venv\Scripts\python.exe") {
    $Py = (Resolve-Path ".venv\Scripts\python.exe").Path
} else {
    $Py = "python"
}

function Step([string]$Message) {
    Write-Host ""
    Write-Host ("==== [{0}] {1} ====" -f (Get-Date -Format "HH:mm:ss"), $Message)
}

# Native commands do not throw on a non-zero exit code in PowerShell 5.1, so
# every step is checked explicitly.
function Invoke-Py([string[]]$PyArgs, [string]$What) {
    & $Py @PyArgs
    if ($LASTEXITCODE -ne 0) {
        throw "$What failed (exit code $LASTEXITCODE). Fix the error above, then re-run this script; completed work is kept."
    }
}

if (-not $env:DAISEE_ROOT) {
    Write-Host "DAISEE_ROOT is not set; falling back to paths.dataset_root in $Config"
}
New-Item -ItemType Directory -Force -Path logs, artifacts | Out-Null
$Started = Get-Date

Step "environment"
# Single quotes only: PowerShell 5.1 does not escape embedded double quotes
# when passing an argument to a native program.
$EnvCheck = @'
import sys, torch
print('python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('No CUDA device. Install the CUDA build of PyTorch '
                     '(pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128) '
                     'and check the NVIDIA driver with nvidia-smi.')
p = torch.cuda.get_device_properties(0)
print(f'gpu: {p.name}  {p.total_memory / 2**30:.1f} GiB')
'@
Invoke-Py @("-c", $EnvCheck) "environment check"

Step "1/6  fetch MediaPipe models"
Invoke-Py @("scripts/fetch_models.py") "model download"

Step "2/6  audit the dataset (layout, labels, subject disjointness)"
Invoke-Py @("scripts/audit_dataset.py", "--config", $Config) "dataset audit"

Step "3/6  Stage 1 preprocessing across $Shards shards"
$Procs = @()
for ($i = 0; $i -lt $Shards; $i++) {
    $Procs += Start-Process -FilePath $Py -NoNewWindow -PassThru -WorkingDirectory $Repo `
        -ArgumentList @("scripts/run_preprocessing.py", "--config", $Config, "--stage", "1",
                        "--shard", "$i/$Shards") `
        -RedirectStandardOutput "logs/stage1_shard$i.out" `
        -RedirectStandardError "logs/stage1_shard$i.err"
}
$Procs | Wait-Process
# Shard exit codes are not trusted individually: the consolidating pass below
# decides against the tolerance using the whole corpus.
Write-Host "shards finished; consolidating (retries any failed clip once more)"
Invoke-Py @("scripts/run_preprocessing.py", "--config", $Config, "--stage", "1") "Stage 1 consolidation"

Step "4/6  fit MRS calibration on the training split"
Invoke-Py @("scripts/fit_mrs_stats.py", "--config", $Config) "MRS calibration"

Step "5/6  GPU memory probe (informational)"
& $Py scripts/probe_finetune_memory.py --config $Config
if ($LASTEXITCODE -ne 0) {
    Write-Host "memory probe failed - continuing; training uses the configured batch size"
}

Step "6/6  fine-tune and evaluate (run: $Run)"
Invoke-Py @("scripts/run_finetune.py", "--config", $Config, "--run-name", $Run,
            "--workers", $Workers, "--resume") "fine-tuning"

Step "figures"
# "--baseline=" rather than --baseline "": PowerShell 5.1 drops an empty-string
# argument to a native program, which would leave --baseline without a value.
& $Py scripts/make_finetune_figures.py --run $Run --baseline=
if ($LASTEXITCODE -ne 0) {
    Write-Host "figures failed - results above are unaffected; re-run: python scripts/make_finetune_figures.py --run $Run --baseline="
}

Step "worked examples (three test clips, three people)"
# Training and evaluation are finished and saved by now, so a failure here is
# reported but does not fail the run.
& $Py scripts/explain_finetuned.py --config $Config --run $Run
if ($LASTEXITCODE -ne 0) {
    Write-Host "worked examples failed - results above are unaffected; re-run: python scripts/explain_finetuned.py --config $Config --run $Run"
}

Step "done"
$Elapsed = (Get-Date) - $Started
Write-Host ("elapsed: {0:N1} h" -f $Elapsed.TotalHours)
Write-Host "report  : artifacts/finetune_report_$Run.json"
Write-Host "figures : artifacts/report_$Run/"
Write-Host "examples: artifacts/examples_$Run/  (face images - do not commit or share)"
Write-Host "history : logs/training_history_$Run.csv"
