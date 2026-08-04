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

    [ValidateSet('Auto', 'Service', 'ScheduledTask')]
    [string]$HostMode = 'Auto',

    [string]$TaskUser = '',

    [switch]$SkipFirewall,
    [switch]$SkipStart
)

$ErrorActionPreference = 'Stop'
$serviceName = 'MSABPMaidBell'
$taskName = 'MSABPMaidBell'
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
$bellConfig = Join-Path $env:ProgramData 'MSABP Maid Bell\bell.json'
$taskLog = Join-Path $RepoRoot "logs\maid-bell.$DeviceId.task.log"
foreach ($requiredPath in @($bellCli, $serviceHost, $maidEntrypoint)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required Maid Bell file does not exist: $requiredPath"
    }
}

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

function Wait-MaidBell {
    $pingHost = if ($ListenHost -eq '0.0.0.0') {
        '127.0.0.1'
    } else {
        $ListenHost
    }
    foreach ($attempt in 1..20) {
        & $script:PythonPath $script:bellCli ping `
            --host $pingHost --port $script:Port 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Invoke-MaidPython -Arguments @(
                $bellCli,
                'ping',
                '--host', $pingHost,
                '--port', [string]$Port
            )
            return
        }
        Start-Sleep -Milliseconds 500
    }
    throw "Maid Bell did not answer on ${pingHost}:$Port within 10 seconds."
}

function Remove-MaidBellTask {
    $existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($null -ne $existingTask) {
        if ($existingTask.State -eq 'Running') {
            Stop-ScheduledTask -TaskName $taskName
        }
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    }
}

function Remove-MaidBellService {
    $existingService = Get-Service -Name $serviceName -ErrorAction SilentlyContinue
    if ($null -eq $existingService) {
        return
    }
    if ($existingService.Status -ne 'Stopped') {
        Stop-Service -Name $serviceName -Force
        $existingService.WaitForStatus('Stopped', [TimeSpan]::FromSeconds(20))
    }
    Invoke-MaidPython -Arguments @($serviceHost, 'remove')
}

function Install-MaidBellService {
    Remove-MaidBellTask
    Invoke-MaidPython -Arguments @(
        '-c',
        'import servicemanager, win32service, win32serviceutil'
    )
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
    if (-not $SkipStart) {
        Invoke-MaidPython -Arguments @(
            $serviceHost,
            '--wait', '20',
            'start'
        )
        Wait-MaidBell
    }
}

function Install-MaidBellTask {
    Remove-MaidBellService
    $resolvedTaskUser = $TaskUser
    if ([string]::IsNullOrWhiteSpace($resolvedTaskUser)) {
        $resolvedTaskUser = "$env:COMPUTERNAME\telecom"
    }
    $actionArguments = (
        '"{0}" serve --config "{1}" --log "{2}"' -f `
            $bellCli, $bellConfig, $taskLog
    )
    $action = New-ScheduledTaskAction `
        -Execute $PythonPath `
        -Argument $actionArguments `
        -WorkingDirectory $RepoRoot
    $trigger = New-ScheduledTaskTrigger -AtStartup
    $principal = New-ScheduledTaskPrincipal `
        -UserId $resolvedTaskUser `
        -LogonType S4U `
        -RunLevel Highest
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -RestartCount 10 `
        -RestartInterval (New-TimeSpan -Minutes 1) `
        -ExecutionTimeLimit ([TimeSpan]::Zero)
    $definition = New-ScheduledTask `
        -Action $action `
        -Trigger $trigger `
        -Principal $principal `
        -Settings $settings
    Register-ScheduledTask `
        -TaskName $taskName `
        -InputObject $definition `
        -Force | Out-Null
    if (-not $SkipStart) {
        Start-ScheduledTask -TaskName $taskName
        try {
            Wait-MaidBell
        } catch {
            $taskInfo = Get-ScheduledTaskInfo -TaskName $taskName
            throw (
                "Scheduled Maid Bell did not start; LastTaskResult=" +
                "$($taskInfo.LastTaskResult). $($_.Exception.Message)"
            )
        }
    }
}

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

if ($HostMode -eq 'Service') {
    Install-MaidBellService
    $selectedHostMode = 'Service'
} elseif ($HostMode -eq 'ScheduledTask') {
    Install-MaidBellTask
    $selectedHostMode = 'ScheduledTask'
} else {
    try {
        Install-MaidBellService
        $selectedHostMode = 'Service'
    } catch {
        Write-Warning (
            'Windows service hosting failed; falling back to an at-startup ' +
            "S4U scheduled task. Original error: $($_.Exception.Message)"
        )
        Remove-MaidBellService
        Install-MaidBellTask
        $selectedHostMode = 'ScheduledTask'
    }
}

if ($selectedHostMode -eq 'Service') {
    Get-Service -Name $serviceName | Format-Table Name, Status, StartType -AutoSize
} else {
    Get-ScheduledTask -TaskName $taskName |
        Format-Table TaskName, State -AutoSize
}
Write-Host "Maid Bell installation/update completed via $selectedHostMode."
Write-Host 'Run the distributed dry-run before any real CST case.'
