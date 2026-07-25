@echo off
rem Double-click this file to start glassprint.
rem
rem It sets itself up the first time (a couple of minutes), then prints two
rem addresses: one for this computer, and one to type into an iPad or phone on
rem the same Wi-Fi. Leave the window open while you work; close it to stop.

cd /d "%~dp0.."

where python >nul 2>&1
if errorlevel 1 (
  echo Python 3 is not installed. Get it from python.org, tick "Add to PATH",
  echo then try again.
  echo.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\glassprint.exe" (
  echo First run - installing glassprint. This takes a minute or two...
  python -m venv .venv
  if errorlevel 1 goto failed
  ".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
  ".venv\Scripts\python.exe" -m pip install --quiet -e .
  if errorlevel 1 goto failed
  echo Done.
  echo.
)

".venv\Scripts\glassprint.exe" serve --lan
echo.
echo glassprint has stopped.
pause
exit /b 0

:failed
echo.
echo Install failed - see the messages above.
pause
exit /b 1
