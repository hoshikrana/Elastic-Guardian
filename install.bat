@echo off
REM Elastic-Guardian X - Windows Installation Script
REM Save this as: install.bat
REM Run from: elastic-guardian-x-complete directory

echo ============================================
echo  Elastic-Guardian X Installation
echo ============================================
echo.

REM Check if we're in the right directory
if not exist "setup.py" (
    echo ERROR: setup.py not found!
    echo Please run this script from the elastic-guardian-x-complete directory
    echo.
    pause
    exit /b 1
)

echo [1/5] Checking Python version...
python --version
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.10 or higher.
    pause
    exit /b 1
)
echo.

echo [2/5] Upgrading pip, setuptools, and wheel...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo WARNING: pip upgrade failed, but continuing...
)
echo.

echo [3/5] Installing PyTorch (CPU version - lighter and faster)...
echo This may take 2-3 minutes...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo ERROR: PyTorch installation failed!
    echo Trying alternative method...
    pip install torch torchvision
)
echo.

echo [4/5] Installing core dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Core dependencies installation failed!
    pause
    exit /b 1
)
echo.

echo [5/5] Installing development tools...
pip install pytest black mypy flake8 isort ruff
echo.

echo Installing package in editable mode...
pip install -e .
if errorlevel 1 (
    echo ERROR: Package installation failed!
    pause
    exit /b 1
)
echo.

echo ============================================
echo  Installation Complete! Testing...
echo ============================================
echo.

echo Testing package import...
python -c "import elastic_guardian; print('✓ Package version:', elastic_guardian.__version__)"
if errorlevel 1 (
    echo ERROR: Package import failed!
    pause
    exit /b 1
)
echo.

echo Testing pytest...
pytest --version
if errorlevel 1 (
    echo WARNING: pytest not found
)
echo.

echo Testing black...
black --version
if errorlevel 1 (
    echo WARNING: black not found
)
echo.

echo ============================================
echo  SUCCESS! Installation complete.
echo ============================================
echo.
echo You can now:
echo   - Run tests: pytest
echo   - Format code: black .
echo   - Type check: mypy elastic_guardian/
echo.
echo Ready to continue with development!
echo.
pause
