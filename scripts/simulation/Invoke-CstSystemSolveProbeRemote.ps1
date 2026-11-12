[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$PythonPath,
    [Parameter(Mandatory = $true)]
    [string]$ProjectPath,
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[a-f0-9]{64}$')]
    [string]$ExpectedProjectSha256,
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[a-f0-9]{64}$')]
    [string]$SourceArchiveSha256,
    [Parameter(Mandatory = $true)]
    [string]$EvidenceRoot,
    [Parameter(Mandatory = $true)]
    [string]$TaskName,
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[a-f0-9]{32}$')]
    [string]$RequestId,
    [ValidateRange(60, 7200)]
    [int]$TimeoutSeconds = 900
)

$ErrorActionPreference = 'Stop'

function Get-RunningCstProcess {
    $knownInfrastructure = @{
        'CSTDCSolverServer_AMD64.exe' = 'C:\Program Files (x86)\CST Studio Suite 2025\AMD64\DC Solver Server\CSTDCSolverServer_AMD64.exe'
        'CSTDCMainController_AMD64.exe' = 'C:\Program Files (x86)\CST Studio Suite 2025\AMD64\DC Main Controller\CSTDCMainController_AMD64.exe'
        'cstd.exe' = 'C:\Program Files (x86)\CST Studio Suite 2025\License Manager\cstd.exe'
    }
    @(
        Get-CimInstance Win32_Process | Where-Object {
            $_.Name -match '(?i)cst|designenvironment|dbstorage|solver'
        } | Where-Object {
            -not (
                $knownInfrastructure.ContainsKey([string]$_.Name) -and
                [string]$_.ExecutablePath -ieq $knownInfrastructure[[string]$_.Name]
            )
        }
    )
}

function Convert-ProcessEvidence($Processes) {
    @(
        $Processes | ForEach-Object {
            [ordered]@{
                name = $_.Name
                pid = [int]$_.ProcessId
                parent_pid = [int]$_.ParentProcessId
                session_id = [int]$_.SessionId
                executable_path = $_.ExecutablePath
                command_line = $_.CommandLine
            }
        }
    )
}

foreach ($value in @($PythonPath, $ProjectPath, $EvidenceRoot, $TaskName, $RequestId)) {
    if ($value.Contains('"') -or $value.Contains("`r") -or $value.Contains("`n") -or
        $value.Contains([char]0)) {
        throw 'A solve-probe value contains a forbidden control or quoting character.'
    }
}
if (-not (Test-Path -LiteralPath $EvidenceRoot -PathType Container)) {
    throw "The preflight evidence root is missing: $EvidenceRoot"
}
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Maid Python is missing: $PythonPath"
}
if (-not (Test-Path -LiteralPath $ProjectPath -PathType Leaf)) {
    throw "CST template is missing: $ProjectPath"
}
$sourceBefore = (Get-FileHash -Algorithm SHA256 -LiteralPath $ProjectPath).Hash.ToLowerInvariant()
if ($sourceBefore -cne $ExpectedProjectSha256) {
    throw 'CST template differs from the approved read-only preflight hash.'
}
$existing = @(Get-RunningCstProcess)
if ($existing.Count) {
    $summary = ($existing | ForEach-Object { $_.Name + ':' + $_.ProcessId }) -join ', '
    throw "A CST-related process exists immediately before solve task registration: $summary"
}

$probePath = Join-Path $EvidenceRoot 'probe_cst_system_solve.py'
$sourceArchivePath = Join-Path $EvidenceRoot 'source_bundle.zip'
$workspace = Join-Path $EvidenceRoot 'workspace'
$reportPath = Join-Path $workspace 'report.json'
if (-not (Test-Path -LiteralPath $probePath -PathType Leaf)) {
    throw 'The verified solve probe source is missing from the evidence root.'
}
if (-not (Test-Path -LiteralPath $sourceArchivePath -PathType Leaf)) {
    throw 'The verified solve source archive is missing from the evidence root.'
}
$archiveBefore = (Get-FileHash -Algorithm SHA256 -LiteralPath $sourceArchivePath).Hash.ToLowerInvariant()
if ($archiveBefore -cne $SourceArchiveSha256) {
    throw 'Solve source archive differs from its approved SHA-256.'
}
if (Test-Path -LiteralPath $workspace) {
    throw 'The isolated solve workspace already exists; refusing to reuse it.'
}

$scheduler = New-Object -ComObject Schedule.Service
$scheduler.Connect()
$folder = $scheduler.GetFolder('\')
try {
    $null = $folder.GetTask($TaskName)
    throw "Task name is not fresh: $TaskName"
} catch [System.IO.FileNotFoundException] {
    # Expected: the one-shot task must not exist before this invocation.
} catch [System.Runtime.InteropServices.COMException] {
    if ($_.Exception.HResult -ne -2147024894) { throw }
}

$definition = $scheduler.NewTask(0)
$definition.RegistrationInfo.Description = 'One-shot real CST solve probe as SYSTEM; no retry and no forced termination.'
$definition.Principal.UserId = 'SYSTEM'
$definition.Principal.LogonType = 5
$definition.Principal.RunLevel = 1
$settings = $definition.Settings
$settings.Enabled = $true
$settings.AllowDemandStart = $true
$settings.StartWhenAvailable = $false
$settings.DisallowStartIfOnBatteries = $false
$settings.StopIfGoingOnBatteries = $false
$settings.RunOnlyIfNetworkAvailable = $false
$settings.RunOnlyIfIdle = $false
$settings.WakeToRun = $false
$settings.MultipleInstances = 2
$settings.ExecutionTimeLimit = 'PT0S'
$settings.AllowHardTerminate = $false
$action = $definition.Actions.Create(0)
$action.Path = $PythonPath
$action.Arguments = (
    '-I -B "' + $probePath +
    '" --project "' + $ProjectPath +
    '" --workspace "' + $workspace +
    '" --report "' + $reportPath +
    '" --expected-project-sha256 ' + $ExpectedProjectSha256 +
    ' --source-archive "' + $sourceArchivePath +
    '" --source-archive-sha256 ' + $SourceArchiveSha256
)
$action.WorkingDirectory = $EvidenceRoot
$task = $folder.RegisterTaskDefinition(
    $TaskName, $definition, 18, 'SYSTEM', $null, 5,
    'O:BAG:BAD:P(A;;GA;;;SY)(A;;GA;;;BA)'
)

$finished = $false
$removed = $false
$taskResult = $null
$watch = [Diagnostics.Stopwatch]::StartNew()
$runRequestedAt = Get-Date
try {
    $null = $task.Run($null)
    while ($watch.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        Start-Sleep -Milliseconds 500
        $current = $folder.GetTask($TaskName)
        $instanceCount = [int]$current.GetInstances(0).Count
        $taskState = [int]$current.State
        $lastRunTime = [datetime]$current.LastRunTime
        $hasRun = $lastRunTime -ge $runRequestedAt.AddSeconds(-2)
        if ($instanceCount -eq 0 -and (Test-Path -LiteralPath $reportPath -PathType Leaf)) {
            $taskResult = [int64]$current.LastTaskResult
            $finished = $true
            break
        }
        if ($instanceCount -eq 0 -and $taskState -notin @(2, 4) -and $hasRun) {
            $taskResult = [int64]$current.LastTaskResult
            $finished = $true
            break
        }
    }
} finally {
    $current = $folder.GetTask($TaskName)
    $instanceCount = [int]$current.GetInstances(0).Count
    if ($instanceCount -eq 0) {
        if ($null -eq $taskResult) { $taskResult = [int64]$current.LastTaskResult }
        $folder.DeleteTask($TaskName, 0)
        $removed = $true
    }
}

if (-not $finished) {
    $sourceAfterTimeout = (Get-FileHash -Algorithm SHA256 -LiteralPath $ProjectPath).Hash.ToLowerInvariant()
    $archiveAfterTimeout = (Get-FileHash -Algorithm SHA256 -LiteralPath $sourceArchivePath).Hash.ToLowerInvariant()
    $residual = @(Get-RunningCstProcess)
    $timeoutResult = [ordered]@{
        schema = 'msabp.cst-system-solve-probe-invocation'
        schema_version = 1
        request_id = $RequestId
        task_name = $TaskName
        task_removed = $removed
        task_result = $taskResult
        task_state = [int]$current.State
        task_running = ([int]$current.GetInstances(0).Count -gt 0)
        timed_out = $true
        evidence_root = $EvidenceRoot
        source_project_before_sha256 = $sourceBefore
        source_project_after_sha256 = $sourceAfterTimeout
        source_project_unchanged = ($sourceAfterTimeout -ceq $ExpectedProjectSha256)
        source_archive_before_sha256 = $archiveBefore
        source_archive_after_sha256 = $archiveAfterTimeout
        source_archive_unchanged = ($archiveAfterTimeout -ceq $SourceArchiveSha256)
        residual_processes = Convert-ProcessEvidence $residual
        elapsed_seconds = [Math]::Round($watch.Elapsed.TotalSeconds, 3)
        success = $false
        report = $null
    }
    $timeoutResult | ConvertTo-Json -Depth 12 |
        Set-Content -LiteralPath (Join-Path $EvidenceRoot 'invocation.json') -Encoding UTF8
    $timeoutResult | ConvertTo-Json -Depth 12 -Compress
    throw "SYSTEM CST solve probe did not finish in $TimeoutSeconds seconds. The running task and all evidence were preserved; no process was terminated."
}

$report = $null
if (Test-Path -LiteralPath $reportPath -PathType Leaf) {
    $report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
}
$sourceAfter = (Get-FileHash -Algorithm SHA256 -LiteralPath $ProjectPath).Hash.ToLowerInvariant()
$archiveAfter = (Get-FileHash -Algorithm SHA256 -LiteralPath $sourceArchivePath).Hash.ToLowerInvariant()
$residualWatch = [Diagnostics.Stopwatch]::StartNew()
do {
    $residual = @(Get-RunningCstProcess)
    if ($residual.Count -eq 0) { break }
    Start-Sleep -Seconds 1
} while ($residualWatch.Elapsed.TotalSeconds -lt 30)
$success = (
    $null -ne $report -and
    $taskResult -eq 0 -and
    $removed -and
    $sourceAfter -ceq $ExpectedProjectSha256 -and
    $archiveAfter -ceq $SourceArchiveSha256 -and
    $residual.Count -eq 0 -and
    $report.success -eq $true
)
$result = [ordered]@{
    schema = 'msabp.cst-system-solve-probe-invocation'
    schema_version = 1
    request_id = $RequestId
    task_name = $TaskName
    task_removed = $removed
    task_result = $taskResult
    timed_out = $false
    evidence_root = $EvidenceRoot
    source_project_before_sha256 = $sourceBefore
    source_project_after_sha256 = $sourceAfter
    source_project_unchanged = ($sourceAfter -ceq $ExpectedProjectSha256)
    source_archive_before_sha256 = $archiveBefore
    source_archive_after_sha256 = $archiveAfter
    source_archive_unchanged = ($archiveAfter -ceq $SourceArchiveSha256)
    residual_process_grace_seconds = [Math]::Round($residualWatch.Elapsed.TotalSeconds, 3)
    residual_processes = Convert-ProcessEvidence $residual
    elapsed_seconds = [Math]::Round($watch.Elapsed.TotalSeconds, 3)
    success = $success
    report = $report
}
$invocationPath = Join-Path $EvidenceRoot 'invocation.json'
$result | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $invocationPath -Encoding UTF8
$result | ConvertTo-Json -Depth 12 -Compress
if (-not $success) { exit 1 }
