[CmdletBinding()]
param(
    [ValidateSet('COCONUTG2')]
    [string]$Target = 'COCONUTG2',
    [string]$PythonPath = 'C:\Users\telecom\miniforge3\envs\maid\python.exe',
    [string]$ProjectPath = 'D:\Academic\Proj_MSABP_2\simulations\models\msa-bp.cst',
    [ValidatePattern('^[a-f0-9]{64}$')]
    [string]$ExpectedProjectSha256 = 'd1962a2d16f976b7d1e4e9eb6d44baf1a460ae324897b4b2c892c0109715d3a8',
    [ValidateRange(60, 7200)]
    [int]$TimeoutSeconds = 1800,
    [switch]$Execute,
    [switch]$PreflightOnly
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$probeSource = Join-Path $PSScriptRoot 'probe_cst_system_solve.py'
$remoteHelperSource = Join-Path $PSScriptRoot 'Invoke-CstSystemSolveProbeRemote.ps1'
foreach ($path in @($probeSource, $remoteHelperSource)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Solve probe source does not exist: $path"
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

function Get-StringSha256([string]$Value) {
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [Text.UTF8Encoding]::new($false).GetBytes($Value)
        return ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace('-', '').ToLowerInvariant()
    } finally {
        $sha.Dispose()
    }
}

function Get-FileSha256([string]$Path) {
    $sha = [Security.Cryptography.SHA256]::Create()
    $stream = [IO.File]::OpenRead($Path)
    try {
        return ([BitConverter]::ToString($sha.ComputeHash($stream))).Replace('-', '').ToLowerInvariant()
    } finally {
        $stream.Dispose()
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
        throw 'A remote solve-probe value contains a forbidden control or quoting character.'
    }
}

function Get-SolveSourceFiles([string]$Root) {
    $files = @(
        Get-ChildItem -LiteralPath (Join-Path $Root 'scripts\automation') -Filter '*.py' -File
        Get-ChildItem -LiteralPath (Join-Path $Root 'scripts\geometry') -Filter '*.py' -File
        Get-Item -LiteralPath (Join-Path $Root 'src\msabp_opt\__init__.py')
        Get-Item -LiteralPath (Join-Path $Root 'src\msabp_opt\simulation\__init__.py')
        Get-ChildItem -LiteralPath (Join-Path $Root 'src\msabp_opt\simulation\distributed') -Filter '*.py' -File
    ) | Sort-Object FullName -Unique
    if ($files.Count -lt 10) { throw 'The fixed solve source closure is unexpectedly small.' }
    return $files
}

function Get-RelativeSourcePath([string]$Root, [string]$Path) {
    $prefix = $Root.TrimEnd('\') + '\'
    $full = [IO.Path]::GetFullPath($Path)
    if (-not $full.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Source file escaped repository root: $full"
    }
    return $full.Substring($prefix.Length).Replace('\', '/')
}

function Get-SolveSourceRecords([string]$Root, [IO.FileInfo[]]$Files) {
    @($Files | ForEach-Object {
        [ordered]@{
            path = Get-RelativeSourcePath $Root $_.FullName
            size_bytes = [int64]$_.Length
            sha256 = Get-FileSha256 $_.FullName
        }
    })
}

function New-SolveSourceBundle(
    [string]$Root,
    [IO.FileInfo[]]$Files,
    [string]$BundlePath
) {
    if (Test-Path -LiteralPath $BundlePath) { throw "Fresh bundle path already exists: $BundlePath" }
    $null = [IO.Directory]::CreateDirectory((Split-Path $BundlePath -Parent))
    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $records = Get-SolveSourceRecords $Root $Files
    $manifest = [ordered]@{
        schema = 'msabp.cst-system-solve-source'
        schema_version = 1
        files = $records
    }
    $archive = [IO.Compression.ZipFile]::Open($BundlePath, [IO.Compression.ZipArchiveMode]::Create)
    try {
        $epoch = [DateTimeOffset]::new(1980, 1, 1, 0, 0, 0, [TimeSpan]::Zero)
        foreach ($file in $Files) {
            $relative = Get-RelativeSourcePath $Root $file.FullName
            $entry = $archive.CreateEntry($relative, [IO.Compression.CompressionLevel]::Optimal)
            $entry.LastWriteTime = $epoch
            $source = [IO.File]::OpenRead($file.FullName)
            $destination = $entry.Open()
            try { $source.CopyTo($destination) } finally { $destination.Dispose(); $source.Dispose() }
        }
        $manifestEntry = $archive.CreateEntry('source_manifest.json', [IO.Compression.CompressionLevel]::Optimal)
        $manifestEntry.LastWriteTime = $epoch
        $manifestBytes = [Text.UTF8Encoding]::new($false).GetBytes(
            ($manifest | ConvertTo-Json -Depth 6) + "`n"
        )
        $manifestStream = $manifestEntry.Open()
        try { $manifestStream.Write($manifestBytes, 0, $manifestBytes.Length) } finally { $manifestStream.Dispose() }
    } finally {
        $archive.Dispose()
    }
    return [pscustomobject]@{
        path = $BundlePath
        sha256 = Get-FileSha256 $BundlePath
        size_bytes = (Get-Item -LiteralPath $BundlePath).Length
        manifest_sha256 = Get-StringSha256 (($manifest | ConvertTo-Json -Depth 6) + "`n")
        file_count = $records.Count
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
$sourceFiles = Get-SolveSourceFiles $repositoryRoot
$sourceRecords = Get-SolveSourceRecords $repositoryRoot $sourceFiles
$sourceClosureSha256 = Get-StringSha256 ($sourceRecords | ConvertTo-Json -Depth 5 -Compress)

$requestId = [Guid]::NewGuid().ToString('N')
$taskName = "ShoreCstSystemSolve-$requestId"
$remoteUploadRoot = "C:\Users\telecom\Downloads\shore-cst-system-solve-$requestId"
$remoteEvidenceRoot = "C:\ProgramData\ShorePrincess\project-msabp\diagnostics\cst-system-solve-$requestId"
$localBundleRoot = Join-Path ([IO.Path]::GetTempPath()) "msabp-cst-system-solve-$requestId"
$localBundlePath = Join-Path $localBundleRoot 'source_bundle.zip'
$probeSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $probeSource).Hash.ToLowerInvariant()
$remoteHelperSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $remoteHelperSource).Hash.ToLowerInvariant()

$plan = [ordered]@{
    schema = 'msabp.cst-system-solve-probe-plan'
    schema_version = 1
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
    source_file_count = $sourceRecords.Count
    source_closure_sha256 = $sourceClosureSha256
    source_bundle_path_on_execute = $localBundlePath
    starts_cst = $true
    runs_solver = $true
    standard_default_tree_case = $true
    required_artifacts = @('S11.csv', 'Rad_Eff.csv', 'Tot_Eff.csv', 'Farfield Source [1].ffs', 'manifest.json')
    preserves_evidence = $true
    force_terminates_processes = $false
    automatic_retry = $false
    read_only_preflight = [bool]$PreflightOnly
}

if ($Execute -and $PreflightOnly) { throw 'Choose either -Execute or -PreflightOnly, not both.' }
if (-not $Execute -and -not $PreflightOnly) {
    $plan | ConvertTo-Json -Depth 6
    Write-Host 'PLAN ONLY. -Execute starts one real CST solve as SYSTEM; it never retries.'
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
    -not ($knownInfrastructure.ContainsKey([string]$_.Name) -and
          [string]$_.ExecutablePath -ieq $knownInfrastructure[[string]$_.Name])
})
if ($existing.Count) {
    throw ('A CST-related process already exists: ' + (($existing | ForEach-Object { $_.Name + ':' + $_.ProcessId }) -join ', '))
}
[ordered]@{
    schema = 'msabp.cst-system-solve-probe-read-only-preflight'
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
Write-Host 'Running the read-only solve preflight. No directory, task, process, or solver is created.'
Invoke-ShoreCommand $plink $identity $node $fingerprint $audit
if ($PreflightOnly) {
    Write-Host 'READ-ONLY PREFLIGHT COMPLETE. CST and the solver were not started.'
    exit 0
}

Write-Host 'Building the fixed, hashed Python source closure for this one-shot solve.'
$bundle = New-SolveSourceBundle $repositoryRoot $sourceFiles $localBundlePath
Write-Host ("Source bundle: {0} bytes, SHA-256 {1}" -f $bundle.size_bytes, $bundle.sha256)

$prepare = @'
$ErrorActionPreference = 'Stop'
$uploadRoot = '__UPLOAD_ROOT__'
$evidenceRoot = '__EVIDENCE_ROOT__'
foreach ($path in @($uploadRoot, $evidenceRoot)) {
    if (Test-Path -LiteralPath $path) { throw "Fresh solve-probe path already exists: $path" }
}
$null = New-Item -ItemType Directory -Path $uploadRoot
$null = New-Item -ItemType Directory -Path $evidenceRoot
[ordered]@{
    schema = 'msabp.cst-system-solve-probe-staging'
    schema_version = 1
    upload_root = $uploadRoot
    evidence_root = $evidenceRoot
} | ConvertTo-Json -Compress
'@
$prepare = $prepare.Replace('__UPLOAD_ROOT__', $remoteUploadRoot).
    Replace('__EVIDENCE_ROOT__', $remoteEvidenceRoot)
Write-Host 'Preparing fresh remote solve-probe staging. CST is not started in this step.'
Invoke-ShoreCommand $plink $identity $node $fingerprint $prepare

$remoteUploadSpec = '{0}@{1}:{2}/' -f $node.UserName, $node.Address, $remoteUploadRoot.Replace('\', '/')
$pscpArguments = @(
    '-batch', '-sftp', '-4', '-P', ([string]$node.Port), '-noagent',
    '-i', $identity, '-hostkey', $fingerprint,
    $probeSource, $remoteHelperSource, $bundle.path, $remoteUploadSpec
)
Write-Host 'Uploading the hashed solve probe and fixed source closure through the Shore key.'
Invoke-NativeCaptured $pscp $pscpArguments 'PSCP upload (remote staging and evidence preserved)'

$launch = @'
$ErrorActionPreference = 'Stop'
$uploadRoot = '__UPLOAD_ROOT__'
$evidenceRoot = '__EVIDENCE_ROOT__'
$names = @('probe_cst_system_solve.py', 'Invoke-CstSystemSolveProbeRemote.ps1', 'source_bundle.zip')
$expected = @{
    'probe_cst_system_solve.py' = '__PROBE_HASH__'
    'Invoke-CstSystemSolveProbeRemote.ps1' = '__HELPER_HASH__'
    'source_bundle.zip' = '__BUNDLE_HASH__'
}
foreach ($name in $names) {
    $uploaded = Join-Path $uploadRoot $name
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $uploaded).Hash.ToLowerInvariant() -cne $expected[$name]) {
        throw "Uploaded solve-probe file failed SHA-256: $name"
    }
    Copy-Item -LiteralPath $uploaded -Destination (Join-Path $evidenceRoot $name) -ErrorAction Stop
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $evidenceRoot $name)).Hash.ToLowerInvariant() -cne $expected[$name]) {
        throw "Evidence-root solve-probe file failed SHA-256: $name"
    }
}
$helper = Join-Path $evidenceRoot 'Invoke-CstSystemSolveProbeRemote.ps1'
& $helper -PythonPath '__PYTHON__' -ProjectPath '__PROJECT__' `
    -ExpectedProjectSha256 '__PROJECT_HASH__' -EvidenceRoot $evidenceRoot `
    -TaskName '__TASK__' -RequestId '__REQUEST__' -TimeoutSeconds __TIMEOUT__ `
    -SourceArchiveSha256 '__BUNDLE_HASH__'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
'@
$launch = $launch.Replace('__UPLOAD_ROOT__', $remoteUploadRoot).
    Replace('__EVIDENCE_ROOT__', $remoteEvidenceRoot).
    Replace('__PROBE_HASH__', $probeSha256).
    Replace('__HELPER_HASH__', $remoteHelperSha256).
    Replace('__BUNDLE_HASH__', $bundle.sha256).
    Replace('__PYTHON__', $PythonPath).
    Replace('__PROJECT__', $ProjectPath).
    Replace('__PROJECT_HASH__', $ExpectedProjectSha256).
    Replace('__TASK__', $taskName).
    Replace('__REQUEST__', $requestId).
    Replace('__TIMEOUT__', [string]$TimeoutSeconds)

Write-Host 'Starting one approved SYSTEM CST solve on the isolated standard default-tree case. No retry is configured.'
Invoke-ShoreCommand $plink $identity $node $fingerprint $launch
