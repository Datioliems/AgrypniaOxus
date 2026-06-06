param(
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path,
    [string]$OutDir = (Join-Path (Resolve-Path "$PSScriptRoot\..").Path "outputs")
)

$ErrorActionPreference = "Stop"

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$zipPath = Join-Path $OutDir "DrowsyDriverAndroid_submission_$timestamp.zip"
$staging = Join-Path $OutDir "submission_staging"

if (Test-Path $staging) {
    Remove-Item -LiteralPath $staging -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $staging | Out-Null

$items = @(
    "app",
    "docs",
    "research",
    "tools",
    "outputs\Bao_cao_nhap_DrowsyDriverAndroid.docx",
    "outputs\project_verification.md",
    "outputs\project_verification.json",
    "outputs\templates",
    "output\playwright\project-plan.png",
    "output\playwright\project-plan-latest.png",
    "output\playwright\pipeline-docs-comparison.png",
    "output\playwright\is54a-visual-roadmap.png",
    "build.gradle",
    "settings.gradle",
    "gradle.properties",
    "local.properties.template",
    "streamlit_app.py",
    "requirements-streamlit.txt",
    "README.md"
)

foreach ($item in $items) {
    $source = Join-Path $ProjectRoot $item
    if (-not (Test-Path $source)) {
        Write-Host "Skipping missing item: $item" -ForegroundColor Yellow
        continue
    }
    $destination = Join-Path $staging $item
    $destinationParent = Split-Path $destination -Parent
    New-Item -ItemType Directory -Force -Path $destinationParent | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination -Recurse -Force
}

$excludePatterns = @(
    "\build\",
    "\.gradle\",
    "\__pycache__\",
    ".pyc"
)

Get-ChildItem -LiteralPath $staging -Recurse -Force | Where-Object {
    $path = $_.FullName
    $excludePatterns | Where-Object { $path.Contains($_) }
} | Sort-Object FullName -Descending | ForEach-Object {
    Remove-Item -LiteralPath $_.FullName -Recurse -Force
}

if (Test-Path $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}
Compress-Archive -Path (Join-Path $staging "*") -DestinationPath $zipPath -Force
Remove-Item -LiteralPath $staging -Recurse -Force

Write-Host "Created $zipPath"
