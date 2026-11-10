@echo off
rem Launch Gravity Sandbox with the crazy preset (release build).
cd /d "%~dp0"
set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
cargo run --release -- --preset crazy %*
if errorlevel 1 pause
