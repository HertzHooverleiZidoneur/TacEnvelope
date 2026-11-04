[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repoName = 'HertzHooverleiZidoneur/TacEnvelope'
$remoteUrl = 'https://github.com/HertzHooverleiZidoneur/TacEnvelope.git'
$releaseRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path

function Invoke-GitChecked {
    param([string[]]$Arguments)
    & git @Arguments
    if ($LASTEXITCODE -ne 0) { throw "git command failed (exit $LASTEXITCODE)." }
}

function Get-HttpStatus {
    param($Failure)
    if ($null -ne $Failure.Exception.Response) { return [int]$Failure.Exception.Response.StatusCode }
    return 0
}

Push-Location -LiteralPath $releaseRoot
try {
    $credentialLines = "protocol=https`nhost=github.com`nusername=HertzHooverleiZidoneur`n`n" | git credential fill
    if ($LASTEXITCODE -ne 0) { throw 'Please sign in first: git credential-manager github login --force' }
    $credentialMap = @{}
    foreach ($entry in $credentialLines) {
        $pair = $entry -split '=', 2
        if ($pair.Count -eq 2) { $credentialMap[$pair[0]] = $pair[1] }
    }
    if (-not $credentialMap['password']) { throw 'No GitHub credential was found.' }
    $apiHeaders = @{
        Authorization = 'Bearer ' + $credentialMap['password']
        Accept = 'application/vnd.github+json'
        'User-Agent' = 'TacEnvelope-publication'
        'X-GitHub-Api-Version' = '2022-11-28'
    }
    try {
        $identity = Invoke-RestMethod -Uri 'https://api.github.com/user' -Headers $apiHeaders
    } catch {
        throw 'GitHub sign-in could not be verified. Run: git credential-manager github login --force'
    }
    Write-Host "Signed in as $($identity.login)."

    $existingRepo = $null
    try {
        $existingRepo = Invoke-RestMethod -Uri "https://api.github.com/repos/$repoName" -Headers $apiHeaders
    } catch {
        if ((Get-HttpStatus $_) -ne 404) { throw }
    }
    if ($null -eq $existingRepo) {
        throw 'Create the empty repository first: https://github.com/new?name=TacEnvelope'
    }
    if ($existingRepo.private) { throw 'The existing destination is private. Review the repository visibility and Pages plan before continuing.' }

    if (-not (Test-Path -LiteralPath '.git')) {
        Invoke-GitChecked -Arguments @('init', '-b', 'main')
    }
    $currentBranch = & git branch --show-current
    if ($currentBranch -ne 'main') { throw 'Expected the local main branch. Review the checkout before publishing.' }
    Invoke-GitChecked -Arguments @('config', 'credential.https://github.com.username', 'HertzHooverleiZidoneur')
    $currentRemote = & git remote get-url origin 2>$null
    if ($LASTEXITCODE -ne 0) {
        Invoke-GitChecked -Arguments @('remote', 'add', 'origin', $remoteUrl)
    } elseif ($currentRemote -ne $remoteUrl) {
        throw 'The local origin points to a different repository. No changes were pushed.'
    }

    $remoteHead = & git ls-remote origin HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Cannot read the destination repository.' }
    if ($remoteHead) {
        $localHead = & git rev-parse --verify HEAD 2>$null
        if ($LASTEXITCODE -ne 0) { throw 'The remote already contains history. Clone it separately and review before publishing.' }
        Invoke-GitChecked -Arguments @('fetch', 'origin', 'main')
        & git merge-base --is-ancestor origin/main HEAD
        if ($LASTEXITCODE -ne 0) { throw 'The remote contains changes not present locally. Review and merge them before publishing.' }
    }
    Invoke-GitChecked -Arguments @('add', '--all')
    & git diff --cached --quiet
    if ($LASTEXITCODE -eq 1) {
        Invoke-GitChecked -Arguments @('commit', '-m', 'Publish TacEnvelope code, paper, video, and project website')
    } elseif ($LASTEXITCODE -ne 0) {
        throw 'Unable to inspect staged changes.'
    }
    Invoke-GitChecked -Arguments @('push', '-u', 'origin', 'main')

    $pagesInfo = $null
    try {
        $pagesInfo = Invoke-RestMethod -Uri "https://api.github.com/repos/$repoName/pages" -Headers $apiHeaders
    } catch {
        if ((Get-HttpStatus $_) -ne 404) { throw }
    }
    if ($null -eq $pagesInfo) {
        $pagesInfo = Invoke-RestMethod -Method Post -Uri "https://api.github.com/repos/$repoName/pages" -Headers $apiHeaders -ContentType 'application/json' -Body '{"build_type":"workflow"}'
    } elseif ($pagesInfo.build_type -ne 'workflow') {
        throw 'The existing Pages site uses a different publishing source. Review it before switching to GitHub Actions.'
    }
    $dispatchBody = '{"ref":"main"}'
    $dispatched = $false
    for ($attempt = 0; $attempt -lt 5; $attempt++) {
        try {
            Invoke-RestMethod -Method Post -Uri "https://api.github.com/repos/$repoName/actions/workflows/pages.yml/dispatches" -Headers $apiHeaders -ContentType 'application/json' -Body $dispatchBody | Out-Null
            $dispatched = $true
            break
        } catch {
            if ((Get-HttpStatus $_) -ne 404 -or $attempt -eq 4) { throw }
            Start-Sleep -Seconds 2
        }
    }
    if (-not $dispatched) { throw 'The files were pushed, but the Pages workflow could not be started.' }
    Write-Host "Repository: https://github.com/$repoName"
    Write-Host "Pages configured at: $($pagesInfo.html_url)"
    Write-Host "Deployment started. Verify success at: https://github.com/$repoName/actions"
} finally {
    $apiHeaders = $null
    $credentialMap = $null
    $credentialLines = $null
    Pop-Location
}
