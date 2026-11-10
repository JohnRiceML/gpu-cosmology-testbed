@echo off
rem Launch Gravity Sandbox with the normal preset (release build).
cd /d "%~dp0"
set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
cargo run --release -- --preset normal %*
if errorlevel 1 pause
