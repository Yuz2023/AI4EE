#!/bin/bash
# Complete setup script that creates any missing files and validates everything

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Complete Multi-LLM Testing Setup     ${NC}"
echo -e "${CYAN}========================================${NC}"
echo

# Function to check if file exists
check_file() {
    if [ -f "$1" ]; then
        echo -e "  ${GREEN}✓${NC} $1 exists"
        return 0
    else
        echo -e "  ${YELLOW}!${NC} $1 missing - creating..."
        return 1
    fi
}

# Function to create directory
create_dir() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
        echo -e "  ${GREEN}✓${NC} Created directory: $1"
    else
        echo -e "  ${GREEN}✓${NC} Directory exists: $1"
    fi
}

echo -e "${BLUE}Step 1: Checking core files...${NC}"

# Check if core files exist (these should already be present)
CORE_FILES=("model_manager.py" "bulk_tester.py" "llm_cli.py")
for file in "${CORE_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "  ${RED}✗${NC} CRITICAL: $file is missing!"
        echo -e "  ${RED}${NC} Please ensure you have all core files from the streamlined version"
        exit 1
    else
        echo -e "  ${GREEN}✓${NC} $file exists"
    fi
done

echo
echo -e "${BLUE}Step 2: Creating directories...${NC}"

# Create all required directories
DIRS=("results" "dataset_cache" "sampled_datasets" "logs")
for dir in "${DIRS[@]}"; do
    create_dir "$dir"
done

echo
echo -e "${BLUE}Step 3: Creating/updating requirements.txt...${NC}"

if ! check_file "requirements.txt"; then
    cat > requirements.txt << 'EOF'
# Core ML Frameworks
torch>=2.5.0
transformers>=4.46.0
vllm>=0.6.4
accelerate>=0.25.0

# Server Dependencies
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
pydantic>=2.9.0
httpx>=0.27.0

# Dataset and Evaluation
promptbench>=0.0.4
datasets>=2.19.0

# Data Processing
numpy>=1.26.0
pandas>=1.3.5
scipy>=1.14.0

# Utilities
tqdm>=4.66.0
requests>=2.32.0
psutil>=5.9.0
pyyaml>=6.0.0
typing_extensions>=4.12.0

# CLI and Display
click>=8.0.0
rich>=13.0.0

# Optional but Recommended
matplotlib>=3.8.0
seaborn>=0.13.0

# Tokenizers
tokenizers>=0.20.0
sentencepiece>=0.2.0
safetensors>=0.4.5
EOF
    echo -e "  ${GREEN}✓${NC} Created requirements.txt"
fi

echo
echo -e "${BLUE}Step 4: Creating validation script...${NC}"

if ! check_file "validate_setup.py"; then
    cat > validate_setup.py << 'EOF'
#!/usr/bin/env python3
"""Quick validation to ensure setup is complete"""
import os
import sys

print("Validating setup...")
errors = []

# Check files
for f in ['model_manager.py', 'bulk_tester.py', 'llm_cli.py', 'config.yaml']:
    if not os.path.exists(f):
        errors.append(f"Missing: {f}")

# Check directories  
for d in ['results', 'dataset_cache', 'sampled_datasets', 'logs']:
    if not os.path.exists(d):
        errors.append(f"Missing dir: {d}")

# Check imports
try:
    import torch, transformers, fastapi, pandas, yaml, click, rich
    print("✓ Core packages installed")
except ImportError as e:
    errors.append(f"Missing package: {e.name}")

if errors:
    print("✗ Issues found:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("✓ All validations passed!")
EOF
    chmod +x validate_setup.py
    echo -e "  ${GREEN}✓${NC} Created validate_setup.py"
fi

echo
echo -e "${BLUE}Step 5: Creating .gitignore...${NC}"

if ! check_file ".gitignore"; then
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*.pyc
.Python

# Virtual Environment
venv/
env/
ENV/
.venv

# Data directories
results/
dataset_cache/
sampled_datasets/
logs/

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Model files
models/
*.bin
*.pth
*.safetensors
*.ckpt

# Environment
.env
*.log

# Temporary
tmp/
temp/
*.tmp
EOF
    echo -e "  ${GREEN}✓${NC} Created .gitignore"
fi

echo
echo -e "${BLUE}Step 6: Creating environment file (.env)...${NC}"

if ! check_file ".env"; then
    cat > .env << 'EOF'
# GPU Configuration
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Performance
TOKENIZERS_PARALLELISM=false
OMP_NUM_THREADS=1

# Server
LLM_SERVER_HOST=0.0.0.0
LLM_SERVER_PORT=8000
EOF
    echo -e "  ${GREEN}✓${NC} Created .env"
fi

echo
echo -e "${BLUE}Step 7: Creating helper scripts...${NC}"

# Create quick test script
if ! check_file "run_quick_test.sh"; then
    cat > run_quick_test.sh << 'EOF'
#!/bin/bash
echo "Starting quick test..."
python model_manager.py --server &
SERVER_PID=$!
sleep 5
python bulk_tester.py --suite quick_test
kill $SERVER_PID
echo "Test complete! Check results/ directory"
EOF
    chmod +x run_quick_test.sh
    echo -e "  ${GREEN}✓${NC} Created run_quick_test.sh"
fi

# Create server start script
if ! check_file "start_server.sh"; then
    cat > start_server.sh << 'EOF'
#!/bin/bash
echo "Starting inference server..."
nohup python model_manager.py --server > logs/server.log 2>&1 &
echo $! > .server.pid
echo "Server started with PID: $(cat .server.pid)"
echo "Logs: tail -f logs/server.log"
EOF
    chmod +x start_server.sh
    echo -e "  ${GREEN}✓${NC} Created start_server.sh"
fi

# Create server stop script
if ! check_file "stop_server.sh"; then
    cat > stop_server.sh << 'EOF'
#!/bin/bash
if [ -f .server.pid ]; then
    PID=$(cat .server.pid)
    kill $PID 2>/dev/null && echo "Server stopped (PID: $PID)"
    rm .server.pid
else
    pkill -f model_manager.py && echo "Server stopped"
fi
EOF
    chmod +x stop_server.sh
    echo -e "  ${GREEN}✓${NC} Created stop_server.sh"
fi

echo
echo -e "${BLUE}Step 8: Installing Python dependencies...${NC}"

# Check if pip is available
if command -v pip3 &> /dev/null; then
    echo -e "  ${CYAN}Installing packages (this may take a few minutes)...${NC}"
    pip3 install -q --upgrade pip
    
    # Install in batches to handle potential conflicts
    pip3 install -q torch transformers 2>/dev/null || echo -e "  ${YELLOW}!${NC} Some packages may need manual installation"
    pip3 install -q fastapi uvicorn pydantic 2>/dev/null
    pip3 install -q pandas numpy pyyaml requests tqdm 2>/dev/null
    pip3 install -q click rich 2>/dev/null
    
    # Try to install vLLM (may fail without CUDA)
    pip3 install -q vllm 2>/dev/null || echo -e "  ${YELLOW}!${NC} vLLM installation failed (may need CUDA)"
    
    # Try to install promptbench
    pip3 install -q promptbench datasets 2>/dev/null || echo -e "  ${YELLOW}!${NC} PromptBench may need manual installation"
    
    echo -e "  ${GREEN}✓${NC} Dependencies installed (check for warnings above)"
else
    echo -e "  ${RED}✗${NC} pip3 not found - please install Python dependencies manually"
fi

echo
echo -e "${BLUE}Step 9: Final validation...${NC}"

# Run Python validation
python3 << 'EOF'
import sys
import os

errors = []
warnings = []

# Check critical files
critical_files = ['model_manager.py', 'bulk_tester.py', 'llm_cli.py', 'config.yaml', 'Makefile']
for f in critical_files:
    if not os.path.exists(f):
        errors.append(f"Missing critical file: {f}")

# Check directories
for d in ['results', 'dataset_cache', 'sampled_datasets', 'logs']:
    if not os.path.exists(d):
        warnings.append(f"Missing directory: {d}")

# Check critical imports
try:
    import torch
    print(f"  ✓ PyTorch installed: {torch.__version__}")
except ImportError:
    errors.append("PyTorch not installed")

try:
    import transformers
    print(f"  ✓ Transformers installed: {transformers.__version__}")
except ImportError:
    errors.append("Transformers not installed")

try:
    import fastapi
    print(f"  ✓ FastAPI installed")
except ImportError:
    errors.append("FastAPI not installed")

try:
    import vllm
    print(f"  ✓ vLLM installed")
except ImportError:
    warnings.append("vLLM not installed (may need CUDA)")

# Check GPU
try:
    import torch
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        warnings.append("CUDA not available (will use CPU)")
except:
    pass

# Summary
if errors:
    print("\n✗ Critical errors found:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
elif warnings:
    print("\n⚠ Warnings:")
    for w in warnings:
        print(f"  - {w}")
    print("\nSetup complete with warnings. The system should work.")
    sys.exit(0)
else:
    print("\n✓ All checks passed!")
    sys.exit(0)
EOF

VALIDATION_RESULT=$?

echo
echo -e "${CYAN}========================================${NC}"

if [ $VALIDATION_RESULT -eq 0 ]; then
    echo -e "${GREEN}       ✓ SETUP COMPLETE!                ${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo
    echo -e "${GREEN}Your Multi-LLM Testing Pipeline is ready!${NC}"
    echo
    echo -e "${BLUE}Quick Start Commands:${NC}"
    echo "  1. Edit config.yaml with model paths:"
    echo "     ${YELLOW}vim config.yaml${NC}"
    echo
    echo "  2. Start the server:"
    echo "     ${YELLOW}./start_server.sh${NC}"
    echo "     or"
    echo "     ${YELLOW}make server${NC}"
    echo
    echo "  3. Run a quick test:"
    echo "     ${YELLOW}./run_quick_test.sh${NC}"
    echo "     or"
    echo "     ${YELLOW}make quick${NC}"
    echo
    echo "  4. View results:"
    echo "     ${YELLOW}make dashboard${NC}"
    echo
    echo -e "${BLUE}Helper Scripts Created:${NC}"
    echo "  • run_quick_test.sh  - Run a quick benchmark"
    echo "  • start_server.sh    - Start server in background"
    echo "  • stop_server.sh     - Stop the server"
    echo "  • validate_setup.py  - Validate installation"
    echo
    echo -e "${BLUE}Directories Created:${NC}"
    echo "  • results/           - Test results"
    echo "  • dataset_cache/     - Cached datasets"
    echo "  • sampled_datasets/  - Pre-sampled data"
    echo "  • logs/             - Application logs"
    echo
    echo -e "${GREEN}Happy Testing! 🚀${NC}"
else
    echo -e "${RED}       ✗ SETUP INCOMPLETE               ${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo
    echo "Please fix the issues above and run this script again."
    echo
    echo "For manual setup, ensure you have:"
    echo "  1. All Python files with correct names (underscores not hyphens)"
    echo "  2. config.yaml with your model paths"
    echo "  3. All required Python packages installed"
    echo
    echo "Need help? Check the README.md file."
    exit 1
fi