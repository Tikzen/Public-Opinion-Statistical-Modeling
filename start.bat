@echo off
echo ======================================
echo  Rumor Spreading Simulator
echo  One-click start script
echo ======================================

cd /d %~dp0

echo Checking virtual environment...

IF NOT EXIST .venv (
echo Creating virtual environment...
python -m venv .venv
)

call .venv\Scripts\activate

echo Installing dependencies...
pip install -r requirements.txt

echo Starting Streamlit app...
python -m streamlit run Home.py

pause
