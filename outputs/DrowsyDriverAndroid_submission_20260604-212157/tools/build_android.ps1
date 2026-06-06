param(
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path,
    [string]$Task = "assembleDebug"
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

if (Test-Path ".\gradlew.bat") {
    Write-Host "Using Gradle wrapper..." -ForegroundColor Cyan
    & .\gradlew.bat $Task
    exit $LASTEXITCODE
}

if ($null -ne (Get-Command gradle -ErrorAction SilentlyContinue)) {
    Write-Host "Using system Gradle..." -ForegroundColor Cyan
    & gradle $Task
    exit $LASTEXITCODE
}

Write-Host "Gradle is not available." -ForegroundColor Yellow
Write-Host "Open this project in Android Studio first, or install Gradle/add a Gradle wrapper."
exit 1
