@echo off
rem Double-click this file to start glassprint.
rem
rem It sets itself up the first time (a minute or two), then prints two
rem addresses: one for this computer, and one to type into an iPad or phone on
rem the same Wi-Fi. Leave the window open while you work; close it to stop.

setlocal
cd /d "%~dp0.."

rem Find a Python that actually runs. The "py" launcher is the dependable one
rem on Windows. Plain "python" is often the Microsoft Store placeholder, which
rem opens the Store and does nothing, so test that it reports a version rather
rem than trusting that the name exists.
set "PY="

py -3 --version >nul 2>&1
if not errorlevel 1 set "PY=py -3"
if defined PY goto have_python

python --version >nul 2>&1
if not errorlevel 1 set "PY=python"
if defined PY goto have_python

echo Could not find Python 3.
echo.
echo Install it from python.org and tick "Add python.exe to PATH" during
echo setup, then run this again. If you installed Python from the Microsoft
echo Store, install it from python.org instead - the Store version cannot
echo create the environment this needs.
echo.
pause
exit /b 1

:have_python
if exist ".venv\Scripts\python.exe" goto run

echo First run - installing glassprint. This takes a minute or two...
%PY% -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
".venv\Scripts\python.exe" -m pip install --quiet -e .
if errorlevel 1 goto failed
echo Done.
echo.

:run
rem Run the module rather than the installed shortcut: one less thing that has
rem to have been generated correctly.
".venv\Scripts\python.exe" -m glassprint.cli serve --lan
echo.
echo glassprint has stopped.
pause
exit /b 0

:failed
echo.
echo Install failed - the messages above say why.
echo.
pause
exit /b 1
