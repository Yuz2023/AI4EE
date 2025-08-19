#!/bin/bash
# Quick Setup Script for Multi-LLM Testing Pipeline
# Run this once to set everything up!

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Banner
echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║         Multi-LLM Testing Pipeline Setup             ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Function to print colored messages
print_status() { echo -e "${GREEN}[✓]${NC} $1"; }
print_error() { echo -e "${RED}[✗]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
print_info() { echo -e "${CYAN}[i]${NC} $1"; }

# Check Python version
print_info "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    print_status "Python $python_version found (>= $required_version required)"
else
    print_error "Python $python_version is too old. Please install Python >= $required_version"
    exit 1
fi

# Check for CUDA/GPU
print_info "Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    print_status "GPU found: $gpu_info"
    cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    print_status "CUDA Version: $cuda_version"
else
    print_warning "No NVIDIA GPU detected. CPU-only mode will be used."
    read -p "Continue with CPU-only setup? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create directory structure
print_info "Creating directory structure..."
mkdir -p results dataset_cache sampled_datasets logs
print_status "Directories created"

# Install Python dependencies
print_info "Installing Python dependencies..."
pip3 install -q --upgrade pip

# Core dependencies
pip3 install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 2>/dev/null || \
    pip3 install -q torch torchvision torchaudio

# Install other dependencies
pip3 install -q \
    transformers \
    vllm \
    fastapi \
    uvicorn \
    promptbench \
    pandas \
    numpy \
    tqdm \
    pyyaml \
    rich \
    click \
    requests \
    matplotlib \
    seaborn

print_status "Dependencies installed"

# Download required scripts if not present
print_info "Checking required scripts..."
required_files=(
    "model_manager.py"
    "bulk_tester.py"
    "llm_cli.py"
    "config.yaml"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -gt 0 ]; then
    print_warning "Missing files: ${missing_files[*]}"
    print_info "Creating default files..."
    
    # Create default config.yaml if missing
    if [[ " ${missing_files[@]} " =~ " config.yaml " ]]; then
        cat > config.yaml << 'EOF'
# Multi-LLM Testing Configuration
models:
  llama-70b:
    path: "/path/to/your/llama-3.1-70b-Instruct"  # UPDATE THIS PATH
    type: "llama"
    tensor_parallel_size: 8
    max_memory_per_gpu: "80GB"
    default_sampling:
      temperature: 0.1
      top_p: 0.9
      max_tokens: 512

datasets:
  quick_test:
    - name: "sst2"
      samples: 100
      metrics: ["accuracy", "valid_format_rate"]
    - name: "bool_logic"
      samples: 100
      metrics: ["accuracy", "valid_format_rate"]
  
  full_benchmark:
    - name: "sst2"
      samples: 1000
      metrics: ["accuracy", "valid_format_rate", "latency"]
    - name: "mmlu"
      samples: 500
      metrics: ["accuracy", "valid_format_rate"]
    - name: "bool_logic"
      samples: 500
      metrics: ["accuracy", "valid_format_rate"]

server:
  host: "0.0.0.0"
  port: 8000
  timeout: 600
  batch_size: 32

inference:
  output_dir: "results"
  use_sampled_data: true

logging:
  level: "INFO"
  file: "inference.log"
EOF
        print_status "Created default config.yaml"
    fi
fi

# Create convenience scripts
print_info "Creating convenience scripts..."

# Create run_test.sh
cat > run_test.sh << 'EOF'
#!/bin/bash
# Quick test runner

echo "Starting inference server..."
python3 model_manager.py --server &
SERVER_PID=$!
sleep 5

echo "Running quick test..."
python3 bulk_tester.py --suite quick_test

echo "Stopping server..."
kill $SERVER_PID

echo "Opening results..."
latest_report=$(ls -t results/report_*.html 2>/dev/null | head -1)
if [ -n "$latest_report" ]; then
    xdg-open "$latest_report" 2>/dev/null || open "$latest_report" 2>/dev/null || echo "View results at: $latest_report"
fi
EOF
chmod +x run_test.sh
print_status "Created run_test.sh"

# Create run_benchmark.sh
cat > run_benchmark.sh << 'EOF'
#!/bin/bash
# Full benchmark runner

echo "Starting full benchmark..."
python3 model_manager.py --server &
SERVER_PID=$!
sleep 5

python3 bulk_tester.py --suite full_benchmark

kill $SERVER_PID

echo "Results saved to results/"
EOF
chmod +x run_benchmark.sh
print_status "Created run_benchmark.sh"

# Test installation
print_info "Testing installation..."
python3 -c "
import torch
import transformers
import fastapi
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Transformers: {transformers.__version__}')
" && print_status "Installation verified" || print_error "Installation verification failed"

# Final instructions
echo
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}          Setup Complete! 🎉                           ${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo
echo -e "${CYAN}Next Steps:${NC}"
echo "1. Edit config.yaml with your model paths:"
echo "   ${YELLOW}vim config.yaml${NC}"
echo
echo "2. Run a quick test:"
echo "   ${YELLOW}make quick${NC}"
echo "   or"
echo "   ${YELLOW}./run_test.sh${NC}"
echo
echo "3. Run full benchmark:"
echo "   ${YELLOW}make full${NC}"
echo "   or"
echo "   ${YELLOW}./run_benchmark.sh${NC}"
echo
echo "4. View results dashboard:"
echo "   ${YELLOW}make dashboard${NC}"
echo
echo -e "${CYAN}Available commands:${NC}"
echo "   ${YELLOW}make help${NC}     - Show all commands"
echo "   ${YELLOW}make status${NC}   - Check system status"
echo "   ${YELLOW}make models${NC}   - List available models"
echo "   ${YELLOW}make results${NC}  - Show latest results"
echo
echo -e "${GREEN}Happy Testing! 🚀${NC}"