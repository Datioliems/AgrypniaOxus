param(
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
)

$ErrorActionPreference = "Stop"

function Test-Command($Name) {
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

Write-Host "Checking Android build environment..." -ForegroundColor Cyan
Write-Host "Project: $ProjectRoot"

$javaOk = Test-Command "java"
$gradleOk = Test-Command "gradle"
$sdkCandidates = @(
    $env:ANDROID_HOME,
    $env:ANDROID_SDK_ROOT,
    "$env:LOCALAPPDATA\Android\Sdk"
) | Where-Object { $_ -and (Test-Path $_) }

if ($javaOk) {
    Write-Host "[OK] Java found" -ForegroundColor Green
    java -version
} else {
    Write-Host "[MISSING] Java/JDK not found in PATH" -ForegroundColor Yellow
}

if (Test-Path "$ProjectRoot\gradlew.bat") {
    Write-Host "[OK] Gradle wrapper found" -ForegroundColor Green
} elseif ($gradleOk) {
    Write-Host "[OK] System Gradle found" -ForegroundColor Green
    gradle --version
} else {
    Write-Host "[MISSING] Gradle wrapper/system Gradle not found" -ForegroundColor Yellow
}

if ($sdkCandidates.Count -gt 0) {
    Write-Host "[OK] Android SDK candidate(s):" -ForegroundColor Green
    $sdkCandidates | ForEach-Object { Write-Host "  $_" }
} else {
    Write-Host "[MISSING] Android SDK not found" -ForegroundColor Yellow
}

$localProperties = Join-Path $ProjectRoot "local.properties"
if (Test-Path $localProperties) {
    Write-Host "[OK] local.properties exists" -ForegroundColor Green
} else {
    Write-Host "[INFO] local.properties missing. Android Studio usually creates it automatically." -ForegroundColor Yellow
    Write-Host "       If needed, copy local.properties.template to local.properties and set sdk.dir."
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
if (-not $javaOk) {
    Write-Host "- Install Android Studio or JDK 17, then reopen the terminal."
}
if ($sdkCandidates.Count -eq 0) {
    Write-Host "- Install Android SDK via Android Studio SDK Manager."
}
if (-not (Test-Path "$ProjectRoot\gradlew.bat") -and -not $gradleOk) {
    Write-Host "- Open the project in Android Studio and let it sync, or add a Gradle wrapper."
}
Write-Host "- Put drowsiness_model.tflite in app/src/main/assets after training CNN."
Write-Host "- Run tools/build_android.ps1 when Java/SDK/Gradle are ready."
