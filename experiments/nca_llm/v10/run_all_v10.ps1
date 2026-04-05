# run_all_v10.ps1
# MVE-20260405-01: Run all 4 external task experiments sequentially
# Usage: Start-Process powershell -ArgumentList "-ExecutionPolicy Bypass -File run_all_v10.ps1" -WindowStyle Minimized

$python = "C:\Users\pipe_render\AppData\Local\Programs\Python\Python314\python.exe"
$ScriptDir = $PSScriptRoot
$repoRoot = (Resolve-Path "$PSScriptRoot\..\..\..").Path
$ResultsDir = "$repoRoot\results\nca_llm\v10"
$LogFile = "$ResultsDir\run_all_v10.log"

Set-Location $ScriptDir

# Create results directory
New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null

$StartTime = Get-Date
"MVE-20260405-01: Starting 4 experiments at $StartTime" | Tee-Object -FilePath $LogFile

$experiments = @(
    @{source="grok";    condition="7b_homo"; label="[1/4]"},
    @{source="grok";    condition="7b_het";  label="[2/4]"},
    @{source="chatgpt"; condition="7b_homo"; label="[3/4]"},
    @{source="chatgpt"; condition="7b_het";  label="[4/4]"}
)

foreach ($exp in $experiments) {
    $label = $exp.label
    $source = $exp.source
    $condition = $exp.condition
    $start = Get-Date

    "" | Tee-Object -FilePath $LogFile -Append
    "$label $source x $condition - started at $start" |
        Tee-Object -FilePath $LogFile -Append

    & $python "$ScriptDir\run_nca_v10.py" --source $source --condition $condition 2>&1 |
        Tee-Object -FilePath $LogFile -Append

    $end = Get-Date
    $elapsed = [math]::Round(($end - $start).TotalMinutes, 1)
    "$label $source x $condition - completed in ${elapsed}min" |
        Tee-Object -FilePath $LogFile -Append
}

"" | Tee-Object -FilePath $LogFile -Append
"Running analysis..." | Tee-Object -FilePath $LogFile -Append
& $python "$ScriptDir\analyze_v10.py" 2>&1 |
    Tee-Object -FilePath $LogFile -Append

$EndTime = Get-Date
$TotalMin = [math]::Round(($EndTime - $StartTime).TotalMinutes, 1)
"" | Tee-Object -FilePath $LogFile -Append
"All done at $EndTime (total: ${TotalMin}min)" |
    Tee-Object -FilePath $LogFile -Append

# Commit
Set-Location $repoRoot
git add results/nca_llm/v10/
git commit -m "data: run MVE-20260405-01 external task verification experiment"
git push
"Committed and pushed." | Tee-Object -FilePath $LogFile -Append
