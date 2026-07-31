@echo off
echo ======================================
echo  Deus - Windows Setup
echo ======================================

echo [1/3] Creating Python virtual environment...
python -m venv venv

echo [2/3] Activating virtual environment...
call venv\Scripts\activate

echo [3/3] Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo ======================================
echo  Setup Complete! 🎉
echo.
echo  To run the bot, execute the following:
echo    venv\Scripts\activate
echo    python main.py
echo ======================================
pause
