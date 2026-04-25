$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

# Remove common generated junk files (outside .venv)
$junkFiles = Get-ChildItem -Force -Recurse -File |
    Where-Object {
        $_.FullName -notmatch '\\.venv(\\|$)' -and (
            $_.Extension -in '.pyc', '.pyo', '.tmp', '.log', '.bak', '.old' -or
            $_.Name -in 'startup.out', 'startup.err'
        )
    }

# Remove common cache directories (outside .venv)
$cacheDirs = Get-ChildItem -Force -Recurse -Directory |
    Where-Object {
        $_.FullName -notmatch '\\.venv(\\|$)' -and
        $_.Name -in '__pycache__', '.pytest_cache', '.mypy_cache'
    }

$deleted = New-Object System.Collections.Generic.List[string]

foreach ($file in $junkFiles) {
    Remove-Item -LiteralPath $file.FullName -Force
    $deleted.Add($file.FullName)
}

# Delete deeper cache folders first
foreach ($dir in ($cacheDirs | Sort-Object FullName -Descending)) {
    if (Test-Path -LiteralPath $dir.FullName) {
        Remove-Item -LiteralPath $dir.FullName -Recurse -Force
        $deleted.Add($dir.FullName + " [dir]")
    }
}

if ($deleted.Count -eq 0) {
    Write-Host "No junk files found. Project is already clean."
} else {
    Write-Host "Deleted items:"
    $deleted | ForEach-Object { Write-Host "- $_" }
}
