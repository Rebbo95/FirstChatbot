@echo off
set /p script_path="Enter the absolute path of the folder containing Chat.py: "

REM Check if Python is installed
python --version > nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Python is not installed. Please install Python and try again.
    pause
    exit /b 1
)

REM Install required Python packages
pip install nltk numpy tensorflow transformers > nul 2>&1

REM Change to the specified directory
cd /d "%script_path%"

REM Run the ChatBot Python script
python Chat.py
