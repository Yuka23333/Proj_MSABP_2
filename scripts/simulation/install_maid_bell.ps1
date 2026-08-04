[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$')]
    [string]$DeviceId,

    [ValidateNotNullOrEmpty()]
    [string]$ListenHost = '0.0.0.0',

    [ValidateRange(1, 65535)]
    [int]$Port = 8766,

    [string]$RepoRoot = '',

    [string]$PythonPath = (
        'C:\Users\telecom\miniforge3\envs\maid\python.exe'
    ),

    [switch]$SkipFirewall,
    [switch]$SkipStart
)

$ErrorActionPreference = 'Stop'
$serviceName = 'MSABPMaidBell'
$firewallRuleName = 'MSABPMaidBell-Tailscale'

function Assert-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    $administrator = [Security.Principal.WindowsBuiltInRole]::Administrator
    if (-not $principal.IsInRole($administrator)) {
        throw 'Run this installer from an elevated PowerShell window.'
    }
}

function Invoke-MaidPython {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & $script:PythonPath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Maid Python command failed with exit code $LASTEXITCODE."
    }
}

Assert-Administrator

$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = Join-Path $scriptDirectory '..\..'
}
$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
$PythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
$bellCli = Join-Path $RepoRoot 'scripts\simulation\maid_bell.py'
$serviceHost = Join-Path $RepoRoot 'scripts\simulation\maid_bell_service.py'
$maidEntrypoint = Join-Path $RepoRoot 'scripts\simulation\maid.py'
foreach ($requiredPath in @($bellCli, $serviceHost, $maidEntrypoint)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required Maid Bell file does not exist: $requiredPath"
    }
}

Invoke-MaidPython -Arguments @(
    '-c',
    'import servicemanager, win32service, win32serviceutil'
)
Invoke-MaidPython -Arguments @(
    $bellCli,
    'init-config',
    '--device-id', $DeviceId,
    '--listen-host', $ListenHost,
    '--port', [string]$Port,
    '--repo-root', $RepoRoot,
    '--python-path', $PythonPath,
    '--maid-entrypoint', $maidEntrypoint
)
Invoke-MaidPython -Arguments @($bellCli, 'doctor')

$existingService = Get-Service -Name $serviceName -ErrorAction SilentlyContinue
if ($null -ne $existingService -and $existingService.Status -ne 'Stopped') {
    Invoke-MaidPython -Arguments @(
        $serviceHost,
        '--wait', '20',
        'stop'
    )
}

$serviceAction = if ($null -eq $existingService) { 'install' } else { 'update' }
Invoke-MaidPython -Arguments @(
    $serviceHost,
    '--startup', 'delayed',
    $serviceAction
)

if (-not $SkipFirewall) {
    $oldRule = Get-NetFirewallRule -Name $firewallRuleName -ErrorAction SilentlyContinue
    if ($null -ne $oldRule) {
        $oldRule | Remove-NetFirewallRule
    }
    New-NetFirewallRule `
        -Name $firewallRuleName `
        -DisplayName 'MSABP Maid Bell (Tailscale only)' `
        -Description 'Allow Princess wake requests only from Tailscale IPv4 peers.' `
        -Enabled True `
        -Direction Inbound `
        -Action Allow `
        -Profile Any `
        -Protocol TCP `
        -LocalPort $Port `
        -RemoteAddress '100.64.0.0/10' | Out-Null
}

if (-not $SkipStart) {
    Invoke-MaidPython -Arguments @(
        $serviceHost,
        '--wait', '20',
        'start'
    )
    $pingHost = if ($ListenHost -eq '0.0.0.0') {
        '127.0.0.1'
    } else {
        $ListenHost
    }
    Invoke-MaidPython -Arguments @(
        $bellCli,
        'ping',
        '--host', $pingHost,
        '--port', [string]$Port
    )
}

Get-Service -Name $serviceName | Format-Table Name, Status, StartType -AutoSize
Write-Host 'Maid Bell installation/update completed.'
Write-Host (
    'The service currently uses its Windows SCM Log On account. ' +
    'Run the distributed dry-run before any real CST case.'
)
