#!/bin/bash

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check CUDA availability
check_cuda() {
    if ! command_exists nvidia-smi; then
        echo "Error: NVIDIA driver and CUDA are not installed."
        echo "Please install NVIDIA drivers and CUDA toolkit first."
        exit 1
    fi
    
    echo "Found NVIDIA driver:"
    nvidia-smi
}

# Function to create and activate virtual environment
setup_venv() {
    echo "Setting up virtual environment..."
    if ! command_exists python3; then
        echo "Error: Python 3 is not installed"
        exit 1
    fi
    
    python3 -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
}

# Function to install CUDA-enabled llama-cpp-python
install_llama_cpp() {
    echo "Installing CUDA-enabled llama-cpp-python..."
    
    # Uninstall any existing llama-cpp-python
    pip uninstall -y llama-cpp-python
    
    # Set environment variables for CUDA build
    export LLAMA_CUBLAS=1
    export CMAKE_ARGS="-DLLAMA_CUBLAS=on"
    
    # First try the pre-built wheel
    pip install llama-cpp-python --extra-index-url https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu118 --no-cache-dir
    
    # Verify CUDA support
    python3 -c "from llama_cpp import Llama; print('CUDA Support:', 'cublas' in str(Llama.library_paths()).lower())"
}

# Main setup process
main() {
    echo "Starting setup process..."
    
    # Check CUDA
    check_cuda
    
    # Setup virtual environment
    setup_venv
    
    # Install CUDA-enabled llama-cpp-python
    install_llama_cpp
    
    # Install other requirements
    echo "Installing other requirements..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install flask tqdm requests scipy psutil gputil==1.4.0 kokoro>=0.3.4 soundfile
    
    echo "Setup completed successfully!"
    echo "Please run: source venv/bin/activate"
}

# Run main function
main 