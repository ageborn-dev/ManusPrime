@echo off
:: ManusPrime Automated Setup and Run Script for Windows

:: Set colors for output
set GREEN=[92m
set YELLOW=[93m
set BLUE=[94m
set RED=[91m
set NC=[0m

echo %BLUE%
echo ███╗   ███╗ █████╗ ███╗   ██╗██╗   ██╗███████╗██████╗ ██████╗ ██╗███╗   ███╗███████╗
echo ████╗ ████║██╔══██╗████╗  ██║██║   ██║██╔════╝██╔══██╗██╔══██╗██║████╗ ████║██╔════╝
echo ██╔████╔██║███████║██╔██╗ ██║██║   ██║███████╗██████╔╝██████╔╝██║██╔████╔██║█████╗  
echo ██║╚██╔╝██║██╔══██║██║╚██╗██║██║   ██║╚════██║██╔═══╝ ██╔══██╗██║██║╚██╔╝██║██╔══╝  
echo ██║ ╚═╝ ██║██║  ██║██║ ╚████║╚██████╔╝███████║██║     ██║  ██║██║██║ ╚═╝ ██║███████╗
echo ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚══════╝╚═╝     ╚═╝  ╚═╝╚═╝╚═╝     ╚═╝╚══════╝
echo %NC%
echo %BLUE%Automated Setup and Run Script for Windows%NC%
echo %BLUE%=======================================%NC%
echo.

:: Function to print messages
call :print_message "Checking for Python 3.8+..."

:: Check Python version
python --version >nul 2>&1
if %errorlevel% neq 0 (
    call :print_error "Python is not installed or not in PATH."
    exit /b 1
)

:: Get Python version
for /f "tokens=2" %%I in ('python --version 2^>^&1') do set PYTHON_VERSION=%%I
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set PYTHON_MAJOR=%%a
    set PYTHON_MINOR=%%b
)

if %PYTHON_MAJOR% LSS 3 (
    call :print_error "Python 3.8+ is required. Found version %PYTHON_VERSION%"
    exit /b 1
)

if %PYTHON_MAJOR% EQU 3 (
    if %PYTHON_MINOR% LSS 8 (
        call :print_error "Python 3.8+ is required. Found version %PYTHON_VERSION%"
        exit /b 1
    )
)

call :print_success "Found Python %PYTHON_VERSION%"

:: Setup virtual environment
if exist venv (
    call :print_message "Virtual environment already exists."
) else (
    call :print_message "Creating virtual environment..."
    python -m venv venv
    if %errorlevel% neq 0 (
        call :print_error "Failed to create virtual environment."
        exit /b 1
    )
    call :print_success "Virtual environment created."
)

:: Activate virtual environment
call :print_message "Activating virtual environment..."
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    call :print_error "Failed to activate virtual environment."
    exit /b 1
)
call :print_success "Virtual environment activated."

:: Install requirements
if exist requirements.txt (
    call :print_message "Installing dependencies..."
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        call :print_error "Failed to install dependencies."
        exit /b 1
    )
    call :print_success "Dependencies installed."
) else (
    call :print_error "requirements.txt not found."
    exit /b 1
)

:: Check for .env file
if not exist .env (
    if exist .env.example (
        call :print_warning "No .env file found. Creating from .env.example..."
        copy .env.example .env
        call :print_warning "Please edit .env file to add your API keys before using the application."
    ) else (
        call :print_warning "No .env or .env.example file found. Creating a basic .env file..."
        echo # Add your API keys here > .env
        echo OPENAI_API_KEY= >> .env
        echo ANTHROPIC_API_KEY= >> .env
        echo MISTRAL_API_KEY= >> .env
        call :print_warning "Please edit .env file to add your API keys before using the application."
    )
) else (
    call :print_message "Found .env file."
)

:: Check for config directory and files
if not exist config (
    call :print_message "Creating config directory..."
    mkdir config
)

if not exist config\default.toml (
    if exist config\default.example.toml (
        call :print_message "Creating default config from example..."
        copy config\default.example.toml config\default.toml
    )
)

:: Start the server
call :print_message "Starting ManusPrime server..."
if exist server.py (
    python server.py
) else (
    call :print_error "server.py not found."
    exit /b 1
)

exit /b 0

:: Functions
:print_message
echo %BLUE%[ManusPrime]%NC% %~1
exit /b 0

:print_success
echo %GREEN%[SUCCESS]%NC% %~1
exit /b 0

:print_warning
echo %YELLOW%[WARNING]%NC% %~1
exit /b 0

:print_error
echo %RED%[ERROR]%NC% %~1
exit /b 0