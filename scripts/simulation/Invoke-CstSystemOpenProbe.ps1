[CmdletBinding()]
param(
    [ValidateSet('COCONUTG2')]
    [string]$Target = 'COCONUTG2',
    [string]$PythonPath = 'C:\Users\telecom\miniforge3\envs\maid\python.exe',
    [string]$ProjectPath = 'D:\Academic\Proj_MSABP_2\simulations\models\msa-bp.cst',
    [ValidatePattern('^[a-f0-9]{64}$')]
    [string]$ExpectedProjectSha256 = 'd1962a2d16f976b7d1e4e9eb6d44baf1a460ae324897b4b2c892c0109715d3a8',
    [ValidateRange(30, 600)]
    [int]$TimeoutSeconds = 180,
    [switch]$Execute,
    [switch]$PreflightOnly
)

$ErrorActionPreference = 'Stop'
$probeSource = Join-Path $PSScriptRoot 'probe_cst_system_open.py'
$remoteHelperSource = Join-Path $PSScriptRoot 'Invoke-CstSystemOpenProbeRemote.ps1'
foreach ($path in @($probeSource, $remoteHelperSource)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Probe source does not exist: $path"
    }
}

function Find-ShoreControlRoot {
    $matches = @(
        [IO.DriveInfo]::GetDrives() |
            Where-Object { $_.IsReady -and $_.DriveType -eq [IO.DriveType]::Removable } |
            ForEach-Object { Join-Path $_.RootDirectory.FullName 'ShoreControl' } |
            Where-Object { Test-Path -LiteralPath (Join-Path $_ 'Control.ps1') -PathType Leaf }
    )
    if ($matches.Count -ne 1) {
        throw "Expected exactly one removable ShoreControl key; found $($matches.Count)."
    }
    return $matches[0]
}

function Get-ShoreFingerprint([string]$PublicKey) {
    if ($PublicKey -cnotmatch '^ssh-ed25519 [A-Za-z0-9+/]+={0,2}(?: .*)?$') {
        throw 'The pinned node host key is not a valid Ed25519 public key.'
    }
    $blob = [Convert]::FromBase64String(($PublicKey.Trim() -split ' ')[1])
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        return 'SHA256:' + [Convert]::ToBase64String($sha.ComputeHash($blob)).TrimEnd('=')
    } finally {
        $sha.Dispose()
    }
}

function ConvertTo-EncodedCommand([string]$Script) {
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($Script))
    if ($encoded.Length -gt 24000) {
        throw "Refusing an oversized remote command ($($encoded.Length) characters)."
    }
    return $encoded
}

function Invoke-NativeCaptured(
    [string]$FilePath,
    [string[]]$ArgumentList,
    [string]$FailureLabel
) {
    $previousPreference = $ErrorActionPreference
    try {
        # Windows PowerShell 5.1 promotes native stderr to NativeCommandError.
        # Continue locally so every line and the real exit code are captured.
        $ErrorActionPreference = 'Continue'
        $output = @(& $FilePath @ArgumentList 2>&1)
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousPreference
    }
    $output | ForEach-Object { Write-Output ([string]$_) }
    if ($exitCode -ne 0) {
        throw "$FailureLabel failed with exit code $exitCode."
    }
}

function Assert-ShoreOnline([string]$ControlPath, [string]$NodeName) {
    $arguments = @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $ControlPath,
        '-Target', $NodeName, '-Command', 'exit 0'
    )
    Invoke-NativeCaptured 'powershell.exe' $arguments 'Shore identity/online gate'
}

function Invoke-ShoreCommand(
    [string]$PlinkPath,
    [string]$IdentityPath,
    $Node,
    [string]$Fingerprint,
    [string]$Script
) {
    $encoded = ConvertTo-EncodedCommand $Script
    $remoteCommand = "pwsh -NoLogo -NoProfile -NonInteractive -EncodedCommand $encoded"
    $arguments = @(
        '-batch', '-no-antispoof', '-T', '-ssh', '-4', '-P', ([string]$Node.Port),
        '-noagent', '-a', '-x', '-noshare', '-no-trivial-auth',
        '-i', $IdentityPath, '-hostkey', $Fingerprint,
        '-l', ([string]$Node.UserName), ([string]$Node.Address), $remoteCommand
    )
    Invoke-NativeCaptured $PlinkPath $arguments 'Remote Shore command'
}

function Assert-PlainRemoteValue([string]$Value) {
    if ($Value.Contains("'") -or $Value.Contains('"') -or $Value.Contains("`r") -or
        $Value.Contains("`n") -or $Value.Contains([char]0)) {
        throw 'A remote probe value contains a forbidden control or quoting character.'
    }
}

$shoreRoot = Find-ShoreControlRoot
$control = Join-Path $shoreRoot 'Control.ps1'
$nodePath = Join-Path $shoreRoot ('nodes\' + $Target.ToLowerInvariant() + '.json')
$plink = Join-Path $shoreRoot 'bin\plink.exe'
$pscp = Join-Path $shoreRoot 'bin\pscp.exe'
$identity = Join-Path $shoreRoot 'identity\controller.ppk'
foreach ($path in @($control, $nodePath, $plink, $pscp, $identity)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required Shore key file is missing: $path"
    }
}

foreach ($binary in @($plink, $pscp)) {
    $signature = Get-AuthenticodeSignature -LiteralPath $binary
    if ($signature.Status -ne [Management.Automation.SignatureStatus]::Valid -or
        $signature.SignerCertificate.Subject -cnotmatch '^CN=Simon Tatham,') {
        throw "The Shore $([IO.Path]::GetFileName($binary)) executable does not have the expected valid publisher signature."
    }
}
$node = Get-Content -LiteralPath $nodePath -Raw | ConvertFrom-Json
if ($node.Version -ne 1 -or $node.HostName -cne $Target -or $node.UserName -cne 'telecom' -or
    [int]$node.Port -ne 22 -or
    $node.Address -cnotmatch '^100\.(?:6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.[0-9]{1,3}\.[0-9]{1,3}$') {
    throw 'The enrolled CoconutG2 node record is outside the approved boundary.'
}
$fingerprint = Get-ShoreFingerprint ([string]$node.HostKey)

$requestId = [Guid]::NewGuid().ToString('N')
$taskName = "ShoreCstSystemOpen-$requestId"
$remoteUploadRoot = "C:\Users\telecom\Downloads\shore-cst-system-open-$requestId"
$remoteEvidenceRoot = "C:\ProgramData\ShorePrincess\project-msabp\diagnostics\cst-system-open-$requestId"
$probeSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $probeSource).Hash.ToLowerInvariant()
$remoteHelperSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $remoteHelperSource).Hash.ToLowerInvariant()

$plan = [ordered]@{
    schema = 'msabp.cst-system-open-probe-plan'
    schema_version = 2
    target = $Target
    control = $control
    node_id = $node.NodeID
    address = $node.Address
    host_key_fingerprint = $fingerprint
    task_name = $taskName
    remote_upload_root = $remoteUploadRoot
    remote_evidence_root = $remoteEvidenceRoot
    python_path = $PythonPath
    project_path = $ProjectPath
    project_sha256 = $ExpectedProjectSha256
    probe_source = $probeSource
    probe_sha256 = $probeSha256
    remote_helper_source = $remoteHelperSource
    remote_helper_sha256 = $remoteHelperSha256
    starts_cst = $true
    runs_solver = $false
    preserves_evidence = $true
    force_terminates_processes = $false
    read_only_preflight = [bool]$PreflightOnly
}

if ($Execute -and $PreflightOnly) {
    throw 'Choose either -Execute or -PreflightOnly, not both.'
}
if (-not $Execute -and -not $PreflightOnly) {
    $plan | ConvertTo-Json -Depth 5
    Write-Host 'PLAN ONLY. Re-run with -Execute only after explicit approval to start CST as SYSTEM.'
    exit 0
}

foreach ($value in @($PythonPath, $ProjectPath, $remoteUploadRoot, $remoteEvidenceRoot, $taskName)) {
    Assert-PlainRemoteValue $value
}

$audit = @'
$ErrorActionPreference = 'Stop'
$python = '__PYTHON__'
$project = '__PROJECT__'
$expectedProject = '__PROJECT_HASH__'
$knownInfrastructure = @{
    'CSTDCSolverServer_AMD64.exe' = 'C:\Program Files (x86)\CST Studio Suite 2025\AMD64\DC Solver Server\CSTDCSolverServer_AMD64.exe'
    'CSTDCMainController_AMD64.exe' = 'C:\Program Files (x86)\CST Studio Suite 2025\AMD64\DC Main Controller\CSTDCMainController_AMD64.exe'
    'cstd.exe' = 'C:\Program Files (x86)\CST Studio Suite 2025\License Manager\cstd.exe'
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw 'Maid Python is missing.' }
if (-not (Test-Path -LiteralPath $project -PathType Leaf)) { throw 'CST template is missing.' }
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $project).Hash.ToLowerInvariant() -cne $expectedProject) {
    throw 'CST template differs from the approved read-only preflight hash.'
}
$cstNamed = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '(?i)cst|designenvironment|dbstorage|solver'
})
$infrastructure = @($cstNamed | Where-Object {
    $knownInfrastructure.ContainsKey([string]$_.Name) -and
    [string]$_.ExecutablePath -ieq $knownInfrastructure[[string]$_.Name]
})
$existing = @($cstNamed | Where-Object {
    -not (
        $knownInfrastructure.ContainsKey([string]$_.Name) -and
        [string]$_.ExecutablePath -ieq $knownInfrastructure[[string]$_.Name]
    )
})
if ($existing.Count) {
    throw ('A CST-related process already exists: ' + (($existing | ForEach-Object { $_.Name + ':' + $_.ProcessId }) -join ', '))
}
[ordered]@{
    schema = 'msabp.cst-system-open-probe-read-only-preflight'
    schema_version = 1
    identity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
    session_id = (Get-Process -Id $PID).SessionId
    source_project_sha256 = $expectedProject
    allowed_infrastructure = @($infrastructure | ForEach-Object {
        [ordered]@{ name = $_.Name; pid = [int]$_.ProcessId; path = $_.ExecutablePath }
    })
} | ConvertTo-Json -Compress
'@
$audit = $audit.Replace('__PYTHON__', $PythonPath).
    Replace('__PROJECT__', $ProjectPath).
    Replace('__PROJECT_HASH__', $ExpectedProjectSha256)

Write-Host 'Checking the Shore node identity and online state.'
Assert-ShoreOnline $control $Target
Write-Host 'Running the read-only CST preflight. No directory, task, or process is created.'
Invoke-ShoreCommand $plink $identity $node $fingerprint $audit
if ($PreflightOnly) {
    Write-Host 'READ-ONLY PREFLIGHT COMPLETE. CST was not started.'
    exit 0
}

$prepare = @'
$ErrorActionPreference = 'Stop'
$uploadRoot = '__UPLOAD_ROOT__'
$evidenceRoot = '__EVIDENCE_ROOT__'
foreach ($path in @($uploadRoot, $evidenceRoot)) {
    if (Test-Path -LiteralPath $path) { throw "Fresh probe path already exists: $path" }
}
$null = New-Item -ItemType Directory -Path $uploadRoot
$null = New-Item -ItemType Directory -Path $evidenceRoot
[ordered]@{
    schema = 'msabp.cst-system-open-probe-staging'
    schema_version = 1
    upload_root = $uploadRoot
    evidence_root = $evidenceRoot
} | ConvertTo-Json -Compress
'@
$prepare = $prepare.Replace('__UPLOAD_ROOT__', $remoteUploadRoot).
    Replace('__EVIDENCE_ROOT__', $remoteEvidenceRoot)
Write-Host 'Preparing fresh remote probe staging (no CST is started in this step).'
Invoke-ShoreCommand $plink $identity $node $fingerprint $prepare

$remoteUploadSpec = '{0}@{1}:{2}/' -f $node.UserName, $node.Address, $remoteUploadRoot.Replace('\', '/')
$pscpArguments = @(
    '-batch', '-sftp', '-4', '-P', ([string]$node.Port), '-noagent',
    '-i', $identity, '-hostkey', $fingerprint,
    $probeSource, $remoteHelperSource, $remoteUploadSpec
)
Write-Host 'Uploading the hashed probe sources through the Shore key.'
Invoke-NativeCaptured $pscp $pscpArguments 'PSCP upload (remote staging and evidence preserved)'

$launch = @'
$ErrorActionPreference = 'Stop'
$uploadRoot = '__UPLOAD_ROOT__'
$evidenceRoot = '__EVIDENCE_ROOT__'
$probeUpload = Join-Path $uploadRoot 'probe_cst_system_open.py'
$helperUpload = Join-Path $uploadRoot 'Invoke-CstSystemOpenProbeRemote.ps1'
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $probeUpload).Hash.ToLowerInvariant() -cne '__PROBE_HASH__') {
    throw 'Uploaded Python probe failed its SHA-256 check.'
}
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $helperUpload).Hash.ToLowerInvariant() -cne '__HELPER_HASH__') {
    throw 'Uploaded remote helper failed its SHA-256 check.'
}
$probe = Join-Path $evidenceRoot 'probe_cst_system_open.py'
$helper = Join-Path $evidenceRoot 'Invoke-CstSystemOpenProbeRemote.ps1'
Copy-Item -LiteralPath $probeUpload -Destination $probe -ErrorAction Stop
Copy-Item -LiteralPath $helperUpload -Destination $helper -ErrorAction Stop
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $probe).Hash.ToLowerInvariant() -cne '__PROBE_HASH__' -or
    (Get-FileHash -Algorithm SHA256 -LiteralPath $helper).Hash.ToLowerInvariant() -cne '__HELPER_HASH__') {
    throw 'Evidence-root source verification failed.'
}
& $helper -PythonPath '__PYTHON__' -ProjectPath '__PROJECT__' `
    -ExpectedProjectSha256 '__PROJECT_HASH__' -EvidenceRoot $evidenceRoot `
    -TaskName '__TASK__' -RequestId '__REQUEST__' -TimeoutSeconds __TIMEOUT__
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
'@
$launch = $launch.Replace('__UPLOAD_ROOT__', $remoteUploadRoot).
    Replace('__EVIDENCE_ROOT__', $remoteEvidenceRoot).
    Replace('__PROBE_HASH__', $probeSha256).
    Replace('__HELPER_HASH__', $remoteHelperSha256).
    Replace('__PYTHON__', $PythonPath).
    Replace('__PROJECT__', $ProjectPath).
    Replace('__PROJECT_HASH__', $ExpectedProjectSha256).
    Replace('__TASK__', $taskName).
    Replace('__REQUEST__', $requestId).
    Replace('__TIMEOUT__', [string]$TimeoutSeconds)

Write-Host 'Starting the approved SYSTEM CST open/close probe. It does not run the solver.'
Invoke-ShoreCommand $plink $identity $node $fingerprint $launch
