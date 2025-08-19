#!/bin/bash

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if conda is installed
check_conda() {
    if ! command -v conda &> /dev/null; then
        print_error "Conda is not installed. Please install Miniconda or Anaconda first."
        echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    print_status "Conda found: $(conda --version)"
}

# Check CUDA availability
check_cuda() {
    if command -v nvidia-smi &> /dev/null; then
        print_status "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
        
        # Get CUDA version
        cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
        print_status "CUDA Version: $cuda_version"
        
        # Check if CUDA version is compatible (12.x recommended)
        if [[ $cuda_version == 12.* ]]; then
            print_status "CUDA 12.x detected - optimal for vLLM"
        else
            print_warning "CUDA $cuda_version detected. CUDA 12.1 is recommended for best compatibility."
        fi
    else
        print_warning "No NVIDIA GPU detected. CPU-only installation will be performed."
        read -p "Continue with CPU-only installation? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Create conda environment
create_environment() {
    ENV_NAME="llama-inference"
    
    # Check if environment already exists
    if conda env list | grep -q "^$ENV_NAME "; then
        print_warning "Environment '$ENV_NAME' already exists."
        read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_status "Removing existing environment..."
            conda env remove -n $ENV_NAME -y
        else
            print_status "Updating existing environment..."
            conda env update -n $ENV_NAME -f environment.yml --prune
            return
        fi
    fi
    
    print_status "Creating conda environment '$ENV_NAME'..."
    conda env create -f environment.yml
    
    if [ $? -eq 0 ]; then
        print_status "Environment created successfully!"
    else
        print_error "Failed to create environment. Check the error messages above."
        exit 1
    fi
}

# Post-installation setup
post_install_setup() {
    print_status "Running post-installation setup..."
    
    # Activate the environment for the setup
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate llama-inference
    
    # Verify key packages
    print_status "Verifying installation..."
    
    python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || print_warning "PyTorch not properly installed"
    python -c "import transformers; print(f'Transformers: {transformers.__version__}')" 2>/dev/null || print_warning "Transformers not properly installed"
    python -c "import vllm; print(f'vLLM: {vllm.__version__}')" 2>/dev/null || print_warning "vLLM not properly installed"
    python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')" 2>/dev/null || print_warning "FastAPI not properly installed"
    
    # Check CUDA availability in PyTorch
    python -c "import torch; print(f'CUDA available in PyTorch: {torch.cuda.is_available()}')"
    python -c "import torch; print(f'CUDA device count: {torch.cuda.device_count()}')" 2>/dev/null
    
    print_status "Creating necessary directories..."
    mkdir -p dataset_cache
    mkdir -p sampled_datasets
    mkdir -p results
    
    print_status "Setup complete!"
}

# Generate activation script
create_activation_script() {
    cat > activate_inference.sh << 'EOF'
#!/bin/bash
# Quick activation script for the inference environment

source $(conda info --base)/etc/profile.d/conda.sh
conda activate llama-inference

echo "Environment activated: llama-inference"
echo "Python version: $(python --version)"
echo "PyTorch CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Set environment variables for optimal performance
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# Show available GPUs
if command -v nvidia-smi &> /dev/null; then
    echo "Available GPUs:"
    nvidia-smi --query-gpu=index,name,memory.free --format=csv
fi

echo ""
echo "Ready to run inference!"
echo "Start the model server with: python model_server_inference.py"
echo "Run bulk inference with: python bulk_inference.py --dataset <dataset_name>"
EOF
    
    chmod +x activate_inference.sh
    print_status "Created activation script: activate_inference.sh"
}

# Main installation flow
main() {
    echo "======================================"
    echo "  LLaMA Inference Environment Setup"
    echo "======================================"
    echo
    
    # Check prerequisites
    print_status "Checking prerequisites..."
    check_conda
    check_cuda
    
    # Check if environment.yml exists
    if [ ! -f "environment.yml" ]; then
        print_error "environment.yml not found in current directory!"
        exit 1
    fi
    
    # Create environment
    create_environment
    
    # Post-installation setup
    post_install_setup
    
    # Create activation script
    create_activation_script
    
    echo
    echo "======================================"
    echo "  Installation Complete!"
    echo "======================================"
    echo
    echo "To activate the environment, run:"
    echo "  source activate_inference.sh"
    echo
    echo "Or manually:"
    echo "  conda activate llama-inference"
    echo
    echo "To start the model server:"
    echo "  python model_server_inference.py"
    echo
    echo "To run inference:"
    echo "  python bulk_inference.py --dataset <dataset_name> --mode parallel"
    echo
}

# Run main function
main