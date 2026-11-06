param(
    [Parameter(Mandatory = $true)][string]$InstallRoot,
    [Parameter(Mandatory = $true)][string]$CheckDirectory,
    [string]$ExpectedVersion = '',
    [int]$TimeoutSeconds = 120,
    [string]$ExpectedWxPythonVersion = ''
)

$ErrorActionPreference = 'Stop'
$installPath = (Resolve-Path -LiteralPath $InstallRoot).Path
$executable = Join-Path $installPath 'bin\stimfit.exe'
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw "No installed application at $executable"
}
New-Item -ItemType Directory -Path $CheckDirectory -Force | Out-Null
$checkPath = (Resolve-Path -LiteralPath $CheckDirectory).Path
$reportPath = Join-Path $checkPath 'snapshot-check.json'
if (Test-Path -LiteralPath $reportPath) { Remove-Item -LiteralPath $reportPath }
$hookPath = Join-Path $installPath 'lib\stimfit\stf_init.py'
$hadHook = Test-Path -LiteralPath $hookPath
if ($hadHook) {
    $originalHook = [IO.File]::ReadAllBytes($hookPath)
    $originalHookTime = (Get-Item -LiteralPath $hookPath).LastWriteTimeUtc
}
$savedEnvironment = @{}
$environmentNames = @('PATH', 'PYTHONHOME', 'PYTHONPATH', 'PYTHONNOUSERSITE',
                      'MPLCONFIGDIR', 'STF_SNAPSHOT_CHECK_DIR',
                      'STF_SNAPSHOT_INSTALL_ROOT', 'STF_SNAPSHOT_EXPECTED_VERSION',
                      'STF_SNAPSHOT_EXPECTED_WXPYTHON_VERSION')
foreach ($name in $environmentNames) {
    $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
$appProcess = $null
try {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'windows_snapshot_check.py') -Destination $hookPath
    $env:PYTHONHOME = $null
    $env:PYTHONPATH = $null
    $env:PYTHONNOUSERSITE = '1'
    $env:MPLCONFIGDIR = Join-Path $checkPath 'matplotlib-config'
    $env:PATH = (Join-Path $installPath 'bin') + ';' + $env:SystemRoot + '\system32;' + $env:SystemRoot
    $env:STF_SNAPSHOT_CHECK_DIR = $checkPath
    $env:STF_SNAPSHOT_INSTALL_ROOT = $installPath
    $env:STF_SNAPSHOT_EXPECTED_VERSION = $ExpectedVersion
    $env:STF_SNAPSHOT_EXPECTED_WXPYTHON_VERSION = $ExpectedWxPythonVersion
    if (-not ('StimfitSnapshotWindowApi' -as [type])) {
        Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class StimfitSnapshotWindowApi {
    [DllImport("user32.dll")]
    public static extern uint GetWindowThreadProcessId(IntPtr hwnd, out uint processId);
    [DllImport("user32.dll", SetLastError=true)]
    public static extern bool PostMessage(IntPtr hwnd, uint message, IntPtr wparam, IntPtr lparam);
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool GetExitCodeProcess(IntPtr process, out uint exitCode);
}
'@
    }
    $appProcess = Start-Process -FilePath $executable -WorkingDirectory $checkPath `
        -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $checkPath 'app-stdout.log') `
        -RedirectStandardError (Join-Path $checkPath 'app-stderr.log')
    $appHandle = $appProcess.Handle
    $timer = [Diagnostics.Stopwatch]::StartNew()
    while (-not (Test-Path -LiteralPath $reportPath)) {
        if ($appProcess.WaitForExit(250)) {
            throw 'Application exited without a snapshot report; inspect app-stdout.log and app-stderr.log'
        }
        if ($timer.Elapsed.TotalSeconds -gt $TimeoutSeconds) {
            throw "Snapshot check exceeded $TimeoutSeconds seconds; inspect the application logs"
        }
    }
    $report = Get-Content -Raw -LiteralPath $reportPath | ConvertFrom-Json
    if (-not $report.window_handle) { throw $report.error }
    [uint32]$windowProcessId = 0
    [StimfitSnapshotWindowApi]::GetWindowThreadProcessId([IntPtr]$report.window_handle, [ref]$windowProcessId) | Out-Null
    if ($windowProcessId -ne $appProcess.Id) { throw 'The close request does not target this test application' }
    [StimfitSnapshotWindowApi]::PostMessage([IntPtr]$report.window_handle, 0x0010, [IntPtr]::Zero, [IntPtr]::Zero) | Out-Null
    if (-not $appProcess.WaitForExit(45000)) { throw 'Application did not exit after closing its main window' }
    [uint32]$applicationExitCode = 259
    if (-not [StimfitSnapshotWindowApi]::GetExitCodeProcess($appHandle, [ref]$applicationExitCode)) {
        throw 'Cannot read the application exit code'
    }
    $report | Add-Member -NotePropertyName exit_code -NotePropertyValue $applicationExitCode
    if ($applicationExitCode -ne 0) {
        $report.passed = $false
        $report | Add-Member -NotePropertyName error -NotePropertyValue "Application exited with code $applicationExitCode" -Force
    }
    $report | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $reportPath -Encoding UTF8
    if (-not $report.passed) { throw $report.error }
    Write-Output ($report | ConvertTo-Json -Depth 6)
} finally {
    if ($appProcess -and -not $appProcess.HasExited) {
        $appProcess.Kill()
        $appProcess.WaitForExit()
    }
    foreach ($name in $environmentNames) {
        [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], 'Process')
    }
    if ($hadHook) {
        [IO.File]::WriteAllBytes($hookPath, $originalHook)
        (Get-Item -LiteralPath $hookPath).LastWriteTimeUtc = $originalHookTime
    } elseif (Test-Path -LiteralPath $hookPath) {
        Remove-Item -LiteralPath $hookPath
    }
}
