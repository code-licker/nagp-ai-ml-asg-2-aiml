@echo off
echo Starting Singapore AI Travel Planning Assistant...
cd /d "%~dp0"
call "..\..\..\.venv\Scripts\activate.bat"
streamlit run app.py
pause
