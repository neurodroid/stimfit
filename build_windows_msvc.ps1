param(
    [switch]$WithPython = $false,
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
Windows workflow. By default this script builds the non-Python patched-biosig preset using
vcpkg dependencies configured through environment variables.

.USAGE
./build_windows_msvc.ps1
./build_windows_msvc.ps1 -WithPython
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
    if ([string]::IsNullOrWhiteSpace($env:VCPKG_ROOT)) {
        $defaultVsVcpkg = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\vcpkg"
        if (Test-Path (Join-Path $defaultVsVcpkg "scripts\buildsystems\vcpkg.cmake")) {
            $env:VCPKG_ROOT = $defaultVsVcpkg
        }
        else {
            throw "VCPKG_ROOT is not set and no bundled Visual Studio vcpkg was found. Set VCPKG_ROOT to your vcpkg checkout."
        }
    }

    if ([string]::IsNullOrWhiteSpace($env:VCPKG_INSTALLED_DIR)) {
        $env:VCPKG_INSTALLED_DIR = Join-Path $repoRoot "vcpkg_installed"
    }

    if ([string]::IsNullOrWhiteSpace($env:VCPKG_BINARY_CACHE_DIR)) {
        $env:VCPKG_BINARY_CACHE_DIR = Join-Path $repoRoot "build\vcpkg-binary-cache"
    }
    if (-not (Test-Path $env:VCPKG_BINARY_CACHE_DIR)) {
        New-Item -ItemType Directory -Path $env:VCPKG_BINARY_CACHE_DIR -Force | Out-Null
    }

    if ([string]::IsNullOrWhiteSpace($env:VCPKG_DEFAULT_BINARY_CACHE)) {
        $env:VCPKG_DEFAULT_BINARY_CACHE = $env:VCPKG_BINARY_CACHE_DIR
    }

    if ([string]::IsNullOrWhiteSpace($env:VCPKG_BINARY_SOURCES)) {
        $env:VCPKG_BINARY_SOURCES = "clear;files,$($env:VCPKG_BINARY_CACHE_DIR),readwrite"
    }

    Write-Host "Using VCPKG_ROOT=$env:VCPKG_ROOT"
    Write-Host "Using VCPKG_INSTALLED_DIR=$env:VCPKG_INSTALLED_DIR"
    Write-Host "Using VCPKG_BINARY_CACHE_DIR=$env:VCPKG_BINARY_CACHE_DIR"
    Write-Host "Using VCPKG_BINARY_SOURCES=$env:VCPKG_BINARY_SOURCES"

    $vcpkgExe = Join-Path $env:VCPKG_ROOT "vcpkg.exe"
    if (-not (Test-Path $vcpkgExe)) {
        throw "Could not find vcpkg executable at '$vcpkgExe'."
    }

    $tripletOverlay = Join-Path $repoRoot "cmake\triplets"
    Invoke-Step -Name "Install vcpkg dependencies" -Command @(
        $vcpkgExe,
        "install",
        "--overlay-triplets", $tripletOverlay,
        "--triplet", "x64-windows-ci-release",
        "--x-install-root", $env:VCPKG_INSTALLED_DIR
    )

    $wxRoot = Join-Path $env:VCPKG_INSTALLED_DIR "x64-windows-ci-release\lib"
    $wxReleaseSetup = Join-Path $wxRoot "mswu\wx\setup.h"
    $wxDebugSetupDir = Join-Path $wxRoot "mswud\wx"
    $wxDebugSetup = Join-Path $wxDebugSetupDir "setup.h"

    if (Test-Path $wxReleaseSetup) {
        if (-not (Test-Path $wxDebugSetupDir)) {
            New-Item -ItemType Directory -Path $wxDebugSetupDir -Force | Out-Null
        }
        Copy-Item -Path $wxReleaseSetup -Destination $wxDebugSetup -Force
        Write-Host "Normalized wx include layout at $wxDebugSetup"
    }

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

    if ([string]::IsNullOrWhiteSpace($selectedBuildPreset)) {
        if ($WithPython) {
            $selectedBuildPreset = "vs2022-release-all-python314-biosig-patched"
        }
        else {
            $selectedBuildPreset = "vs2022-release-all-biosig-patched"
        }
    }

    $resolvedBuildDir = $BuildDir
    if ([string]::IsNullOrWhiteSpace($resolvedBuildDir)) {
        $resolvedBuildDir = Join-Path "..\stimfit-out" $selectedConfigurePreset
    }

    $configureCommand = @("cmake", "--preset", $selectedConfigurePreset)
    if (-not [string]::IsNullOrWhiteSpace($BuildDir)) {
        $configureCommand += @("-B", $resolvedBuildDir)
    }
    Invoke-Step -Name "Configure" -Command $configureCommand

    if ([string]::IsNullOrWhiteSpace($BuildDir)) {
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
