# process_and_train.ps1
# Extracts .npy sequences from every new recording subfolder, then trains.
# Each subfolder inside recordings\ holds individual short clips of one sign.
# The FOLDER NAME becomes the label shown on-screen and used in the model.
#
# Run from the server\ directory:
#   cd d:\projects\full_project\server
#   powershell -ExecutionPolicy Bypass -File scripts\process_and_train.ps1

$ErrorActionPreference = "Stop"

$ScriptsDir  = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ServerRoot  = Split-Path -Parent $ScriptsDir

$RecRoot       = Join-Path $ServerRoot "data\recordings"
$DatasetRoot   = Join-Path $ServerRoot "data\dataset"
$ExtractScript = Join-Path $ScriptsDir "extract_dataset.py"
$TrainScript   = Join-Path $ScriptsDir "train_from_dataset.py"

# Folder name = ASL label shown on screen
$NewFolders = @(
    "Good Bye",
    "Hello",
    "Jury",
    "This Is",
    "Us"
)

Write-Host ""
Write-Host "=== ASL Dataset Extraction + Training Pipeline ===" -ForegroundColor Cyan
Write-Host "  Recordings : $RecRoot"
Write-Host "  Dataset    : $DatasetRoot"
Write-Host ""

$totalSkipped = 0

foreach ($folder in $NewFolders) {
    $folderPath = Join-Path $RecRoot $folder
    if (-not (Test-Path $folderPath -PathType Container)) {
        Write-Host "[SKIP] Folder not found: $folderPath" -ForegroundColor Yellow
        $totalSkipped++
        continue
    }

    $clips = Get-ChildItem $folderPath -Filter "*.mp4" -File | Sort-Object Name
    if ($clips.Count -eq 0) {
        Write-Host "[SKIP] No .mp4 clips in: $folderPath" -ForegroundColor Yellow
        $totalSkipped++
        continue
    }

    $count = $clips.Count
    Write-Host "--- Label: '$folder' ($count clips) ---" -ForegroundColor Green

    $clipNum = 0
    foreach ($clip in $clips) {
        $clipNum++
        Write-Host "  [$clipNum/$count] $($clip.Name)" -ForegroundColor DarkGray

        & python $ExtractScript `
            --video "$($clip.FullName)" `
            --label "$folder" `
            --output "$DatasetRoot" `
            --mode segment `
            --no-display

        if ($LASTEXITCODE -ne 0) {
            Write-Warning "  extract_dataset.py failed (exit $LASTEXITCODE) for: $($clip.Name)"
        }
    }

    $labelDataDir = Join-Path $DatasetRoot $folder
    $savedCount = 0
    if (Test-Path $labelDataDir) {
        $savedCount = (Get-ChildItem $labelDataDir -Filter "*.npy" -ErrorAction SilentlyContinue).Count
    }
    Write-Host "  '$folder' done. Dataset now has $savedCount sequences." -ForegroundColor Green
    Write-Host ""
}

Write-Host "--- Extraction done. Labels skipped: $totalSkipped ---" -ForegroundColor Cyan
Write-Host ""
Write-Host "=== Starting Training ===" -ForegroundColor Cyan
Write-Host "  Dataset root: $DatasetRoot"
Write-Host ""

& python $TrainScript --dataset-root "$DatasetRoot"

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Training failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "=== Training Complete! ===" -ForegroundColor Green
