@echo off
:: =============================================================================
:: run.bat  –  Start Pokémon Blue AI training
:: =============================================================================
:: Edit the two paths below, then double-click this file (or run from CMD).
:: Do NOT launch from inside BizHawk; use this script so the socket flags
:: are passed correctly.
:: =============================================================================

:: --- CONFIGURE THESE --------------------------------------------------------
set BIZHAWK_EXE=E:\Local\Documents\Emulators\BizHawk-2.11-win-x64\EmuHawk.exe
set PYTHON_ARGS=--algo PPO --steps 500000
:: Optional: set your Anthropic API key here if you haven't set it globally
:: set ANTHROPIC_API_KEY=sk-ant-api03-...
:: ----------------------------------------------------------------------------

:: Change to the folder this batch file lives in (the project root)
cd /d "%~dp0"

:: Activate Python virtual environment if one exists
if exist "venv\Scripts\activate.bat" (
    echo [RUN] Activating venv...
    call venv\Scripts\activate.bat
) else (
    echo [RUN] No venv found, using system Python.
)

:: Install / update dependencies if requirements.txt is present
if exist "requirements.txt" (
    echo [RUN] Installing dependencies from requirements.txt...
    pip install -r requirements.txt --quiet
    if errorlevel 1 (
        echo.
        echo [ERROR] pip install failed. Check your internet connection and Python install.
        echo.
        pause
        exit /b 1
    )
    echo [RUN] Dependencies OK.
)

:: Verify Python can find the training module
python -c "import training" 2>nul
if errorlevel 1 (
    echo.
    echo [ERROR] Cannot import training module.
    echo         Make sure you are running this from the project root folder.
    echo.
    pause
    exit /b 1
)

:: Verify BizHawk executable exists
if not exist "%BIZHAWK_EXE%" (
    echo.
    echo [ERROR] BizHawk not found at:
    echo         %BIZHAWK_EXE%
    echo.
    echo         Edit BIZHAWK_EXE at the top of run.bat to fix this.
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Pokemon Blue AI  –  Starting up
echo ============================================================
echo  BizHawk : %BIZHAWK_EXE%
echo  Args    : %PYTHON_ARGS%
echo ============================================================
echo.

:: Step 1: Start Python training in a new window.
::   The training script starts the TCP server on 127.0.0.1:65432 and
::   waits for BizHawk to connect before the training loop begins.
echo [RUN] Starting Python training server...
start "Pokemon AI - Python" cmd /k "python -m training.train %PYTHON_ARGS%"

:: Step 2: Wait until Python is actually listening on port 65432.
::   torch + stable_baselines3 imports can take 15-30 s on first run.
::   We poll netstat every 2 s instead of using a fixed sleep so BizHawk
::   never tries to connect before the socket is ready.
echo [RUN] Waiting for Python to open port 65432...
:wait_loop
netstat -an | find "65432" | find "LISTENING" >nul 2>&1
if errorlevel 1 (
    timeout /t 2 /nobreak >nul
    goto wait_loop
)
echo [RUN] Port 65432 is ready.

:: Step 3: Launch BizHawk with the socket connection flags.
::   --socket_ip / --socket_port tell BizHawk to connect to Python's server.
::   After BizHawk opens, load the Lua script manually:
::     Tools -> Lua Console -> Open Script -> emulator/bizhawk_script.lua -> Run
echo [RUN] Launching BizHawk...
start "Pokemon AI - BizHawk" "%BIZHAWK_EXE%" --socket_ip=127.0.0.1 --socket_port=65432

echo.
echo [RUN] Done. Next steps:
echo   1. In BizHawk: load your Pokemon Blue ROM  (File - Open ROM)
echo   2. Tools - Lua Console
echo   3. Open Script: emulator\bizhawk_script.lua
echo   4. Click Run (the play button)
echo.
echo   The Python window will show "Emulator connected" once the Lua
echo   script is running and training will start automatically.
echo.
pause
