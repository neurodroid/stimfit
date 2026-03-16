param(
    [switch]$WithPython = $true,
    [string]$ConfigurePreset,
    [string]$BuildPreset,
    [string]$PackageGenerator,
    [string]$InstallPrefix,
    [string]$BuildDir
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

<#
.SYNOPSIS
Configure, build, install, and optionally package Stimfit on Windows/MSVC using CMake.

.DESCRIPTION
Analogous to build_macos_cmake.sh, but tailored for the Visual Studio 2022 / MSVC preset-based
Windows workflow. By default this script builds the Python-enabled preset that uses the patched
biosig workflow.

.USAGE
./build_windows_msvc.ps1
./build_windows_msvc.ps1 -WithPython:$false
./build_windows_msvc.ps1 -InstallPrefix ..\stimfit-out\install\custom-python
./build_windows_msvc.ps1 -PackageGenerator INNOSETUP
./build_windows_msvc.ps1 -PackageGenerator ZIP

Optional parameter overrides:
-ConfigurePreset <preset-name>
-BuildPreset <preset-name>
-BuildDir <path-to-configured-build-dir>
#>

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string[]]$Command
    )

    Write-Host "==> $Name" -ForegroundColor Cyan
    Write-Host "    $($Command -join ' ')"

    & $Command[0] $Command[1..($Command.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed ($Name) with exit code $LASTEXITCODE"
    }
}

$repoRoot = Resolve-Path $PSScriptRoot
Push-Location $repoRoot

try {
    $selectedConfigurePreset = $ConfigurePreset
    $selectedBuildPreset = $BuildPreset

    if ([string]::IsNullOrWhiteSpace($selectedConfigurePreset)) {
        if ($WithPython) {
            $selectedConfigurePreset = "vs2022-vcpkg-wx-hdf5-python314-biosig-patched"
        }
        else {
            $selectedConfigurePreset = "vs2022-vcpkg-wx-hdf5-biosig-patched"
        }
    }

    $useBuildPreset = -not [string]::IsNullOrWhiteSpace($selectedBuildPreset)

    $resolvedBuildDir = $BuildDir
    if ([string]::IsNullOrWhiteSpace($resolvedBuildDir)) {
        $resolvedBuildDir = Join-Path "..\stimfit-out" $selectedConfigurePreset
    }

    Invoke-Step -Name "Configure" -Command @("cmake", "--preset", $selectedConfigurePreset)

    if ($useBuildPreset) {
        Invoke-Step -Name "Build" -Command @("cmake", "--build", "--preset", $selectedBuildPreset)
    }
    else {
        Invoke-Step -Name "Build" -Command @("cmake", "--build", $resolvedBuildDir, "--config", "Release", "--target", "ALL_BUILD")
    }

    $installCommand = @("cmake", "--install", $resolvedBuildDir, "--config", "Release")
    if (-not [string]::IsNullOrWhiteSpace($InstallPrefix)) {
        $installCommand += @("--prefix", $InstallPrefix)
    }
    Invoke-Step -Name "Install" -Command $installCommand

    if (-not [string]::IsNullOrWhiteSpace($PackageGenerator)) {
        $resolvedPackageGenerator = $PackageGenerator.ToUpperInvariant()
        if ($resolvedPackageGenerator -notin @("INNOSETUP", "ZIP")) {
            throw "Unsupported package generator '$PackageGenerator'. Expected INNOSETUP or ZIP."
        }

        $packageConfig = Join-Path $resolvedBuildDir "CPackConfig.cmake"
        if (-not (Test-Path $packageConfig)) {
            throw "CPack configuration was not found at '$packageConfig'. Run configure first and verify the build directory."
        }

        $packageCommand = @("cpack", "--config", $packageConfig, "-C", "Release", "-G", $resolvedPackageGenerator)
        Invoke-Step -Name "Package" -Command $packageCommand
    }

    $displayInstallPrefix = $InstallPrefix
    if ([string]::IsNullOrWhiteSpace($displayInstallPrefix)) {
        $displayInstallPrefix = "the preset-defined install prefix"
    }

    Write-Host "Build completed successfully." -ForegroundColor Green
    Write-Host "Configure preset: $selectedConfigurePreset"
    Write-Host "Build preset: $selectedBuildPreset"
    Write-Host "Build directory: $resolvedBuildDir"
    Write-Host "Install prefix: $displayInstallPrefix"
}
finally {
    Pop-Location
}
