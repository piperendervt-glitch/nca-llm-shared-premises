# run_all_v5.ps1
# MVE-20260404-05: Run all 12 experiments sequentially
# Usage (PowerShell): .\run_all_v5.ps1
# Background: Start-Process powershell -ArgumentList "-File run_all_v5.ps1" -WindowStyle Minimized

$python = "C:\Users\pipe_render\AppData\Local\Programs\Python\Python314\python.exe"
$repoRoot = (Resolve-Path "$PSScriptRoot\..\..\..").Path
$script = "$PSScriptRoot\run_nca_v5.py"
$logfile = "$repoRoot\results\nca_llm\v5\run_all_v5.log"

Set-Location $PSScriptRoot

$experiments = @(
    # Priority 1: 7b conditions (most important)
    @{ task = "math_elementary"; condition = "7b_homo" },
    @{ task = "math_elementary"; condition = "7b_het" },
    @{ task = "math_middle";     condition = "7b_homo" },
    @{ task = "math_middle";     condition = "7b_het" },
    @{ task = "math_high";       condition = "7b_homo" },
    @{ task = "math_high";       condition = "7b_het" },
    # Priority 2: 3b conditions (supplementary)
    @{ task = "math_elementary"; condition = "3b_homo" },
    @{ task = "math_elementary"; condition = "3b_het" },
    @{ task = "math_middle";     condition = "3b_homo" },
    @{ task = "math_middle";     condition = "3b_het" },
    @{ task = "math_high";       condition = "3b_homo" },
    @{ task = "math_high";       condition = "3b_het" }
)

$start = Get-Date
Write-Output "MVE-05: Starting $($experiments.Count) experiments at $start" | Tee-Object -FilePath $logfile

for ($i = 0; $i -lt $experiments.Count; $i++) {
    $exp = $experiments[$i]
    $n = $i + 1
    $ts = Get-Date
    Write-Output "" | Tee-Object -FilePath $logfile -Append
    Write-Output "[$n/$($experiments.Count)] $($exp.task) x $($exp.condition) - started at $ts" | Tee-Object -FilePath $logfile -Append

    & $python $script --task $exp.task --condition $exp.condition 2>&1 | Tee-Object -FilePath $logfile -Append

    $te = Get-Date
    $dur = ($te - $ts).TotalMinutes
    Write-Output "[$n/$($experiments.Count)] Completed in $([math]::Round($dur, 1)) min" | Tee-Object -FilePath $logfile -Append
}

$end = Get-Date
$total = ($end - $start).TotalHours
Write-Output "" | Tee-Object -FilePath $logfile -Append
Write-Output "All experiments complete. Total: $([math]::Round($total, 1)) hours" | Tee-Object -FilePath $logfile -Append
