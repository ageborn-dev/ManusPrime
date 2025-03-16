#!/bin/bash
# ManusPrime Automated Setup and Run Script

# Text colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored messages
print_message() {
  echo -e "${BLUE}[ManusPrime]${NC} $1"
}

print_success() {
  echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
  echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Check Python version
check_python_version() {
  local python_cmd=$1
  local python_version=$($python_cmd -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  local major_version=$(echo $python_version | cut -d. -f1)
  local minor_version=$(echo $python_version | cut -d. -f2)
  
  if [ "$major_version" -lt 3 ] || [ "$major_version" -eq 3 -a "$minor_version" -lt 8 ]; then
    return 1
  else
    return 0
  fi
}

# Function to set up virtual environment
setup_venv() {
  if [ -d "venv" ]; then
    print_message "Virtual environment already exists."
  else
    print_message "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    if [ $? -ne 0 ]; then
      print_error "Failed to create virtual environment."
      exit 1
    fi
    print_success "Virtual environment created."
  fi
}

# Function to activate virtual environment
activate_venv() {
  print_message "Activating virtual environment..."
  if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
  else
    # Unix-like
    source venv/bin/activate
  fi
  
  if [ $? -ne 0 ]; then
    print_error "Failed to activate virtual environment."
    exit 1
  fi
  print_success "Virtual environment activated."
}

# Function to install requirements
install_requirements() {
  if [ -f "requirements.txt" ]; then
    print_message "Installing dependencies..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
      print_error "Failed to install dependencies."
      exit 1
    fi
    print_success "Dependencies installed."
  else
    print_error "requirements.txt not found."
    exit 1
  fi
}

# Function to check if .env file exists
check_env_file() {
  if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
      print_warning "No .env file found. Creating from .env.example..."
      cp .env.example .env
      print_warning "Please edit .env file to add your API keys before using the application."
    else
      print_warning "No .env or .env.example file found. Creating a basic .env file..."
      echo "# Add your API keys here" > .env
      echo "OPENAI_API_KEY=" >> .env
      echo "ANTHROPIC_API_KEY=" >> .env
      echo "MISTRAL_API_KEY=" >> .env
      print_warning "Please edit .env file to add your API keys before using the application."
    fi
  else
    print_message "Found .env file."
  fi
}

# Function to create default config if it doesn't exist
check_config() {
  if [ ! -d "config" ]; then
    print_message "Creating config directory..."
    mkdir -p config
  fi
  
  if [ ! -f "config/default.toml" ] && [ -f "config/default.example.toml" ]; then
    print_message "Creating default config from example..."
    cp config/default.example.toml config/default.toml
  fi
}

# Function to start the server
start_server() {
  print_message "Starting ManusPrime server..."
  if [ -f "server.py" ]; then
    python server.py
  else
    print_error "server.py not found."
    exit 1
  fi
}

# Main execution starts here

# Display welcome message
echo -e "${BLUE}"
echo "███╗   ███╗ █████╗ ███╗   ██╗██╗   ██╗███████╗██████╗ ██████╗ ██╗███╗   ███╗███████╗"
echo "████╗ ████║██╔══██╗████╗  ██║██║   ██║██╔════╝██╔══██╗██╔══██╗██║████╗ ████║██╔════╝"
echo "██╔████╔██║███████║██╔██╗ ██║██║   ██║███████╗██████╔╝██████╔╝██║██╔████╔██║█████╗  "
echo "██║╚██╔╝██║██╔══██║██║╚██╗██║██║   ██║╚════██║██╔═══╝ ██╔══██╗██║██║╚██╔╝██║██╔══╝  "
echo "██║ ╚═╝ ██║██║  ██║██║ ╚████║╚██████╔╝███████║██║     ██║  ██║██║██║ ╚═╝ ██║███████╗"
echo "╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚══════╝╚═╝     ╚═╝  ╚═╝╚═╝╚═╝     ╚═╝╚══════╝"
echo -e "${NC}"
echo -e "${BLUE}Automated Setup and Run Script${NC}"
echo -e "${BLUE}==============================${NC}"
echo ""

# Check for Python
print_message "Checking for Python 3.8+..."
PYTHON_CMD=""

# Try python3 first, then python
if command_exists python3; then
  if check_python_version python3; then
    PYTHON_CMD="python3"
  fi
fi

if [ -z "$PYTHON_CMD" ] && command_exists python; then
  if check_python_version python; then
    PYTHON_CMD="python"
  fi
fi

if [ -z "$PYTHON_CMD" ]; then
  print_error "Python 3.8 or higher is required but not found."
  exit 1
fi

python_version=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
print_success "Found Python $python_version."

# Setup and activate virtual environment
setup_venv
activate_venv

# Install dependencies
install_requirements

# Check for .env file
check_env_file

# Check for config
check_config

# Start the server
start_server