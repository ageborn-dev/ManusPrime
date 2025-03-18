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

:: Start the server
call :print_message "Starting ManusPrime server..."
if exist server.py (
    python server.py
) else (
    call :print_error "server.py not found."
    exit /b 1
)