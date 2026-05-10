@echo off
title DS360 Impact — Launcher
color 0A
echo.
echo  ================================
echo   DS360 Impact — Career AI System
echo  ================================
echo.

REM Try common python paths
set PYTHON=

if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\python.exe" (
    set PYTHON=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\python.exe
)
if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311\python.exe" (
    set PYTHON=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311\python.exe
)
if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python313\python.exe" (
    set PYTHON=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python313\python.exe
)
if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python314\python.exe" (
    set PYTHON=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python314\python.exe
)
if exist "C:\Python311\python.exe" set PYTHON=C:\Python311\python.exe
if exist "C:\Python312\python.exe" set PYTHON=C:\Python312\python.exe
if exist "E:\python\python.exe"    set PYTHON=E:\python\python.exe

if "%PYTHON%"=="" (
    where python >nul 2>&1 && set PYTHON=python
)

if "%PYTHON%"=="" (
    echo  [ERROR] Python not found on this machine.
    echo.
    echo  Please install Python 3.11+ from:
    echo  https://www.python.org/downloads/
    echo.
    echo  Then re-run this file.
    pause
    exit /b 1
)

echo  [OK] Python found: %PYTHON%
echo.
echo  [1/2] Installing dependencies...
"%PYTHON%" -m pip install streamlit pandas numpy plotly scikit-learn --quiet --upgrade
echo.
echo  [2/2] Launching DS360 Impact...
echo.
echo  Open in browser: http://localhost:8501
echo.
cd /d "%~dp0"
"%PYTHON%" -m streamlit run app.py --server.port 8501 --server.headless false
pause
