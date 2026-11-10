@echo off
rem Launch Gravity Sandbox: 2D cosmological structure-formation run (periodic comoving PM).
rem Output goes to D:\cosmo-runs\<UTC timestamp>\ (meta.json, pk.csv, snapshots).
cd /d "%~dp0"
set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
cargo run --release -- --preset cosmic %*
if errorlevel 1 pause
