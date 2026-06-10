$ErrorActionPreference = "Stop"

$BinDir = "$PSScriptRoot\bin"
if (-not (Test-Path $BinDir)) {
    New-Item -ItemType Directory -Path $BinDir | Out-Null
}

Write-Host "Checking monitoring binaries..."

# 1. Prometheus
$PromVersion = "2.54.1"
$PromZip = "$BinDir\prometheus.zip"
$PromExe = "$BinDir\prometheus-$PromVersion.windows-amd64\prometheus.exe"
if (-not (Test-Path $PromExe)) {
    Write-Host "Downloading Prometheus..."
    Invoke-WebRequest -Uri "https://github.com/prometheus/prometheus/releases/download/v$PromVersion/prometheus-$PromVersion.windows-amd64.zip" -OutFile $PromZip
    Write-Host "Extracting Prometheus..."
    Expand-Archive -Path $PromZip -DestinationPath $BinDir -Force
    Remove-Item $PromZip
}

# 2. Grafana
$GrafVersion = "11.2.0"
$GrafZip = "$BinDir\grafana.zip"
$GrafExe = "$BinDir\grafana-v$GrafVersion\bin\grafana-server.exe"
if (-not (Test-Path $GrafExe)) {
    Write-Host "Downloading Grafana..."
    Invoke-WebRequest -Uri "https://dl.grafana.com/oss/release/grafana-$GrafVersion.windows-amd64.zip" -OutFile $GrafZip
    Write-Host "Extracting Grafana..."
    Expand-Archive -Path $GrafZip -DestinationPath $BinDir -Force
    Remove-Item $GrafZip
}

# 3. Loki
$LokiVersion = "3.1.1"
$LokiZip = "$BinDir\loki.zip"
$LokiExe = "$BinDir\loki-windows-amd64.exe"
if (-not (Test-Path $LokiExe)) {
    Write-Host "Downloading Loki..."
    Invoke-WebRequest -Uri "https://github.com/grafana/loki/releases/download/v$LokiVersion/loki-windows-amd64.exe.zip" -OutFile $LokiZip
    Write-Host "Extracting Loki..."
    Expand-Archive -Path $LokiZip -DestinationPath $BinDir -Force
    Remove-Item $LokiZip
}

# 4. Promtail
$PromtailZip = "$BinDir\promtail.zip"
$PromtailExe = "$BinDir\promtail-windows-amd64.exe"
if (-not (Test-Path $PromtailExe)) {
    Write-Host "Downloading Promtail..."
    Invoke-WebRequest -Uri "https://github.com/grafana/loki/releases/download/v$LokiVersion/promtail-windows-amd64.exe.zip" -OutFile $PromtailZip
    Write-Host "Extracting Promtail..."
    Expand-Archive -Path $PromtailZip -DestinationPath $BinDir -Force
    Remove-Item $PromtailZip
}

Write-Host "----------------------------------------"
Write-Host "Starting Services in Background..."
Write-Host "----------------------------------------"

# Stop existing instances if running
Stop-Process -Name "prometheus" -ErrorAction SilentlyContinue
Stop-Process -Name "grafana-server" -ErrorAction SilentlyContinue
Stop-Process -Name "loki-windows-amd64" -ErrorAction SilentlyContinue
Stop-Process -Name "promtail-windows-amd64" -ErrorAction SilentlyContinue

# Start Prometheus
Write-Host "Starting Prometheus on port 9090..."
Start-Process -FilePath $PromExe -ArgumentList "--config.file=""$PSScriptRoot\prometheus.yml""" -WindowStyle Hidden

# Start Loki
Write-Host "Starting Loki on port 3100..."
Start-Process -FilePath $LokiExe -ArgumentList "--config.file=""$PSScriptRoot\loki-config.yml""" -WindowStyle Hidden

# Start Promtail
Write-Host "Starting Promtail..."
Start-Process -FilePath $PromtailExe -ArgumentList "--config.file=""$PSScriptRoot\promtail-config.windows.yml""" -WindowStyle Hidden

# Start Grafana
# We need to set GF_PATHS_PROVISIONING to load the dashboards automatically
$env:GF_PATHS_PROVISIONING = "$PSScriptRoot\grafana\provisioning"
$env:GF_SERVER_HTTP_PORT = "5000"
Write-Host "Starting Grafana on port 5000..."
Start-Process -FilePath $GrafExe -WorkingDirectory "$BinDir\grafana-v$GrafVersion" -WindowStyle Hidden

Write-Host ""
Write-Host "✅ All services started!"
Write-Host "Grafana is available at http://localhost:5000 (admin/admin)"
Write-Host "To stop them later, run:"
Write-Host "Stop-Process -Name prometheus,grafana-server,loki-windows-amd64,promtail-windows-amd64"
