@echo off
cd /d "%~dp0"

if not exist ".venv" (
    echo Membuat virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    echo Install dependency ^(sekali saja, perlu beberapa menit^)...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)

echo.
echo Membuka http://localhost:8501
streamlit run app.py
pause
