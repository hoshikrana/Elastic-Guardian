@echo off
REM EGX — Elastic Guardian X — Windows Installer
echo EGX Elastic Guardian X — Windows Install
echo.
python --version >nul 2>&1
IF ERRORLEVEL 1 (echo ERROR: Python not found. Install Python 3.10+ first. & pause & exit /b 1)
echo [1/4] Creating virtual environment...
python -m venv .venv
call .venv\Scripts\activate.bat
echo [2/4] Upgrading pip...
python -m pip install --upgrade pip
echo [3/4] Installing EGX (dev mode)...
pip install -e ".[dev]"
echo [4/4] Verifying...
python -c "import egx; print('EGX', egx.__version__, 'OK')"
echo.
echo Done. Activate: .venv\Scripts\activate.bat
pause
