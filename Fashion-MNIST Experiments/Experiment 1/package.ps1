$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$zip = Join-Path $root 'cvhomework.zip'
if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
$stage = Join-Path ([IO.Path]::GetTempPath()) ('cvhomework-' + [guid]::NewGuid().ToString())
$destRoot = Join-Path $stage 'cvhomework'
New-Item -ItemType Directory -Path $destRoot -Force | Out-Null
$exclude = @('data','checkpoints','logs','.venv','.git','__pycache__','.pytest_cache')
Get-ChildItem -LiteralPath $root -Recurse -File | ForEach-Object {
  $rel = $_.FullName.Substring($root.Length + 1)
  $parts = $rel -split '[\\/]'
  if ($_.Extension -ne '.pyc' -and -not ($parts | Where-Object { $_ -in $exclude }) -and -not ($rel -like 'results/*') -and $rel -ne 'cvhomework.zip') {
    $target = Join-Path $destRoot $rel
    New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
    Copy-Item -LiteralPath $_.FullName -Destination $target
  }
}
Compress-Archive -Path $destRoot -DestinationPath $zip
Remove-Item -LiteralPath $stage -Recurse -Force
Write-Host "Created $zip"
