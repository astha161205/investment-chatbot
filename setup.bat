@echo off
echo Setting up Investment Assistant...

REM Check if virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

echo Setup complete!
echo.
echo To run the application:
echo 1. Activate the environment: venv\Scripts\activate
echo 2. Run the application: python your_script_name.py
echo 3. Open your browser and go to: http://localhost:5000
echo.
echo Or simply run: run.bat

pause 