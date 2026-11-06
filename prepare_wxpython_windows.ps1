param(
    [string]$PythonExecutable,
    [string]$WxPythonVersion = '4.3.2',
    [string]$BuildRoot,
    [string]$JomExecutable,
    [ValidateRange(1, 64)][int]$Jobs = 8
)

# Prepare an isolated wheel and matching native development libraries for
# Stimfit's embedded Python build. Does not change the Python installation.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Invoke-Native {
    param([string]$Executable, [string[]]$Arguments)
    $ErrorActionPreference = 'Continue'
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Executable failed with exit code $LASTEXITCODE"
    }
}

if (-not $PythonExecutable) {
    $PythonExecutable = & py -3.14 -c 'import sys; print(sys.executable)'
    if ($LASTEXITCODE -ne 0) { throw 'Install 64-bit Python 3.14 or pass -PythonExecutable.' }
}
$PythonExecutable = (Resolve-Path -LiteralPath $PythonExecutable).Path
if (-not $BuildRoot) { $BuildRoot = Join-Path $PSScriptRoot "build/wxpython-$WxPythonVersion" }
$BuildRoot = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($BuildRoot)
New-Item -ItemType Directory -Path $BuildRoot -Force | Out-Null
$sitePackages = Join-Path $BuildRoot 'site-packages'
$sourceRoot = Join-Path $BuildRoot "wxpython-$WxPythonVersion"
$wxLibDir = Join-Path $sourceRoot 'ext/wxWidgets/lib/vc140_x64_dll'

@'
import struct, sys
assert sys.platform == 'win32' and struct.calcsize('P') == 8, '64-bit Windows Python is required'
assert sys.version_info[:2] == (3, 14), 'The Windows Python preset requires Python 3.14'
'@ | & $PythonExecutable -
if ($LASTEXITCODE -ne 0) { throw 'Unsupported Python interpreter.' }

$savedProgress = $ProgressPreference
try {
    $ProgressPreference = 'SilentlyContinue'
    $release = Invoke-RestMethod -Uri "https://pypi.org/pypi/wxPython/$WxPythonVersion/json"
    $source = @($release.urls | Where-Object packagetype -eq 'sdist')
    if ($source.Count -ne 1) { throw 'Expected one wxPython source archive.' }
    $archive = Join-Path $BuildRoot $source[0].filename
    if (-not (Test-Path -LiteralPath $archive)) {
        Write-Host "Downloading wxPython $WxPythonVersion sources..."
        Invoke-WebRequest -Uri $source[0].url -OutFile $archive -UseBasicParsing
    }
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $source[0].digests.sha256) {
        throw "Source checksum mismatch: $archive"
    }
} finally {
    $ProgressPreference = $savedProgress
}

Invoke-Native $PythonExecutable @('-m', 'pip', 'install', '--only-binary=:all:',
    '--no-deps', '--upgrade', '--target', $sitePackages, "wxPython==$WxPythonVersion")
if (-not (Test-Path -LiteralPath (Join-Path $sourceRoot 'build.py'))) {
    Invoke-Native 'tar.exe' @('-xf', $archive, '-C', $BuildRoot)
}

$buildArguments = @('build.py', 'build_wx', '--x64', '--release', '--cairo')
$savedPath = $env:PATH
$savedCompilerFlags = $env:_CL_
if ($JomExecutable) {
    $JomExecutable = (Resolve-Path -LiteralPath $JomExecutable).Path
    # Order consumers after each module's PCH producer in the unpacked sources.
    # The upstream nmake file otherwise races when executed by jom.
    @'
from pathlib import Path
import re, sys
root = Path(sys.argv[1]) / 'ext/wxWidgets/build/msw'
makefile = root / 'makefile.vc'
text = makefile.read_text()
marker = '# Stimfit: order precompiled header creation for jom'
if marker not in text:
    rules, stems = [], set()
    for stem, name in re.findall(r'^\$\(OBJS\)\\(\w+dll)_([^:]+)\.obj:', text, re.M):
        stems.add(stem)
        if name != 'dummy':
            rules.append(r'$(OBJS)\%s_%s.obj: $(OBJS)\%s_dummy.obj' % (stem, name, stem))
    if not rules:
        raise SystemExit('Could not identify wxWidgets PCH build rules')
    makefile.write_text(text + '\n' + marker + '\n' + '\n'.join(rules) + '\n')
    for stem in stems:
        (root / 'vc140_x64_mswudll' / (stem + '_dummy.obj')).unlink(missing_ok=True)
'@ | & $PythonExecutable - $sourceRoot
    if ($LASTEXITCODE -ne 0) { throw 'Failed to order wxWidgets PCH build rules.' }
    $env:PATH = (Split-Path -Parent $JomExecutable) + ';' + $env:PATH
    $env:_CL_ = ($savedCompilerFlags + ' /FS').Trim()
    $buildArguments += @('--jom', "--extra_make=-j$Jobs")
}
Push-Location $sourceRoot
try {
    Invoke-Native $PythonExecutable $buildArguments
} finally {
    Pop-Location
    $env:PATH = $savedPath
    $env:_CL_ = $savedCompilerFlags
}

# Link with import libraries from the matching release's wxPython build options,
# and ship the wheel's exact runtime DLLs in both bin/ and the wx package.
Copy-Item -Path (Join-Path $sitePackages 'wx/wx*.dll') -Destination $wxLibDir -Force
@'
import sys
sys.path.insert(0, sys.argv[1])
import wx, wx.aui, wx.py.shell
assert wx.__version__ == sys.argv[2], wx.version()
print(wx.version())
print(wx.__file__)
'@ | & $PythonExecutable - $sitePackages $WxPythonVersion
if ($LASTEXITCODE -ne 0) { throw 'Prepared wxPython failed its import check.' }

Write-Host "wxPython $WxPythonVersion is ready in $BuildRoot"
Write-Host "Python executable: $PythonExecutable"
