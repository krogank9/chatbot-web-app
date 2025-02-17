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

    # Install build dependencies
    pip install --upgrade pip wheel setuptools packaging
    pip install cmake ninja
    
    # Install CUDA build tools
    if ! command_exists nvcc; then
        echo "Installing CUDA toolkit..."
        apt-get update && apt-get install -y cuda-toolkit-11-8
    fi

    # Set environment variables for CUDA build
    export CUDA_HOME=/usr/local/cuda
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    
    export GGML_CUDA=1
    export CMAKE_ARGS="-DGGML_CUDA=ON -DLLAMA_CUBLAS=OFF"
    
    echo "Building llama-cpp-python with CUDA support..."
    pip install llama-cpp-python --no-cache-dir --verbose
    
    # Verify installation
    echo "Verifying installation..."
    if python3 -c "import llama_cpp; print('llama_cpp version:', llama_cpp.__version__)"; then
        echo "llama-cpp-python installed successfully"
    else
        echo "Failed to import llama_cpp. Trying alternative installation..."
        # Try alternative installation method
        pip install --force-reinstall --no-cache-dir llama-cpp-python --prefer-binary
    fi
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
    
    # Final verification
    echo "Performing final verification..."
    if ! python3 -c "import llama_cpp; print('llama_cpp installation verified')"; then
        echo "ERROR: llama-cpp-python installation failed"
        exit 1
    fi
    
    echo "Setup completed successfully!"
    echo "Please run: source venv/bin/activate"
}

# Run main function
main 