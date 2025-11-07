@echo off
setlocal enabledelayedexpansion

REM ================================
REM  Step 1. Locate Anaconda activate.bat
REM ================================
set "foundPath="

echo Searching for Anaconda installation...

REM First try the default C: path
if exist "C:\Anaconda3\Scripts\activate.bat" set "foundPath=C:\Anaconda3\Scripts\activate.bat"
if exist "C:\Anaconda\Scripts\activate.bat" set "foundPath=C:\Anaconda\Scripts\activate.bat"

REM If not found in C:, search all other drives
if not defined foundPath (
    for %%D in (D E F G H I J K L M N O P Q R S T U V W X Y Z) do (
        if exist "%%D:\Anaconda3\Scripts\activate.bat" (
            set "foundPath=%%D:\Anaconda3\Scripts\activate.bat"
            goto :found
        )
        if exist "%%D:\anaconda\Scripts\activate.bat" (
            set "foundPath=%%D:\anaconda\Scripts\activate.bat"
            goto :found
        )
    )
)

:found
if defined foundPath (
    echo Found Anaconda activation script at: %foundPath%
    call "%foundPath%"
    call conda activate base
) else (
    echo Anaconda not found. Using system Python instead.
)

REM ================================
REM  Step 2. Run Streamlit App
REM ================================
cd /d "%~dp0"
echo Launching Streamlit app from: %cd%
streamlit run app.py

endlocal
pause
