param(
    [switch]$SkipPrepare
)

$ErrorActionPreference = "Stop"

function Run-Step {
    param(
        [Parameter(Mandatory=$true)][string]$Name,
        [Parameter(Mandatory=$true)][string[]]$Command
    )

    Write-Host "==> $Name" -ForegroundColor Cyan
    Write-Host "    $($Command -join ' ')"
    & $Command[0] $Command[1..($Command.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed ($Name) with exit code $LASTEXITCODE"
    }
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $repoRoot

try {
    if (-not $SkipPrepare) {
        Run-Step -Name "Prepare patched biosig" -Command @("cmake", "-P", "cmake/PrepareBiosigMSVC.cmake")
    }

    Run-Step -Name "Configure Stimfit preset" -Command @("cmake", "--preset", "vs2022-vcpkg-wx-hdf5-biosig-patched")
    Run-Step -Name "Build Stimfit preset" -Command @("cmake", "--build", "--preset", "vs2022-release-stimfit-biosig-patched")

    Write-Host "All steps completed successfully." -ForegroundColor Green
}
finally {
    Pop-Location
}
