#!/bin/bash
# Automated setup script that fixes everything and sets up the environment

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Multi-LLM Testing Pipeline Auto-Setup ${NC}"
echo -e "${BLUE}========================================${NC}"
echo

# Step 1: Fix file names
echo -e "${GREEN}Step 1: Fixing file names...${NC}"

# Function to safely rename files
safe_rename() {
    if [ -f "$1" ]; then
        if [ ! -f "$2" ]; then
            mv "$1" "$2"
            echo -e "  ${GREEN}✓${NC} Renamed $1 → $2"
        else
            echo -e "  ${YELLOW}!${NC} $2 already exists, skipping"
        fi
    elif [ -f "$2" ]; then
        echo -e "  ${GREEN}✓${NC} $2 already correct"
    else
        echo -e "  ${RED}✗${NC} $1 not found"
    fi
}

# Rename files to correct names
safe_rename "model-manager.py" "model_manager.py"
safe_rename "bulk-tester.py" "bulk_tester.py"
safe_rename "llm-cli.py" "llm_cli.py"
safe_rename "config-yaml.txt" "config.yaml"
safe_rename "simplified-makefile.txt" "Makefile"
safe_rename "readme-streamlined.md" "README.md"

echo

# Step 2: Create directories
echo -e "${GREEN}Step 2: Creating directories...${NC}"
mkdir -p results dataset_cache sampled_datasets logs
echo -e "  ${GREEN}✓${NC} Directories created"
echo

# Step 3: Fix encoding issues
echo -e "${GREEN}Step 3: Fixing character encoding...${NC}"

# Create temporary Python script to fix encoding
cat > fix_encoding_temp.py << 'EOF'
#!/usr/bin/env python3
import os

files = ['model_manager.py', 'bulk_tester.py', 'llm_cli.py', 
         'config.yaml', 'Makefile', 'README.md', 'quick_setup.sh']

replacements = {
    'Ã—': '×', 'â†'': '→', 'â†"': '↓', 'âœ"': '✓', 'âœ—': '✗',
    'ðŸš€': '🚀', 'ðŸ"Š': '📊', 'ðŸ§ª': '🧪', 'ðŸ"‹': '📋',
    'ðŸ"¦': '📦', 'ðŸŽ¯': '🎯', 'ðŸ"§': '🔧', 'ðŸ"': '📁',
    'ðŸŽ›ï¸': '🎛️', 'ðŸ"ˆ': '📈', 'ðŸ›': '🐛', 'ðŸ"': '📝',
    'ðŸ"„': '📄', 'ðŸ¤': '🤝', 'ðŸ†˜': '🆘', 'ðŸŽ‰': '🎉',
}

for filepath in files:
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            for bad, good in replacements.items():
                content = content.replace(bad, good)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"  ✓ Fixed {filepath}")
        except Exception as e:
            print(f"  ✗ Error with {filepath}: {e}")
EOF

python3 fix_encoding_temp.py
rm fix_encoding_temp.py
echo

# Step 4: Create requirements.txt if not exists
echo -e "${GREEN}Step 4: Creating requirements.txt...${NC}"
if [ ! -f "requirements.txt" ]; then
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

# Dataset and Evaluation
promptbench>=0.0.4
datasets>=2.19.0
pandas>=1.3.5
numpy>=1.26.0

# Utilities
pyyaml>=6.0.0
requests>=2.32.0
tqdm>=4.66.0
click>=8.0.0
rich>=13.0.0
EOF
    echo -e "  ${GREEN}✓${NC} requirements.txt created"
else
    echo -e "  ${GREEN}✓${NC} requirements.txt already exists"
fi
echo

# Step 5: Check Python version
echo -e "${GREEN}Step 5: Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo -e "  ${GREEN}✓${NC} Python $python_version (>= $required_version required)"
else
    echo -e "  ${RED}✗${NC} Python $python_version is too old. Please install Python >= $required_version"
    exit 1
fi
echo

# Step 6: Install dependencies
echo -e "${GREEN}Step 6: Installing Python dependencies...${NC}"
echo -e "${YELLOW}This may take a few minutes...${NC}"

# Check if we should use conda or pip
if command -v conda &> /dev/null; then
    echo -e "  ${BLUE}ℹ${NC} Conda detected, using conda environment"
    # You can add conda-specific commands here if needed
fi

# Install with pip
pip3 install -q --upgrade pip
pip3 install -q -r requirements.txt

echo -e "  ${GREEN}✓${NC} Dependencies installed"
echo

# Step 7: Check GPU
echo -e "${GREEN}Step 7: Checking GPU/CUDA...${NC}"
if command -v nvidia-smi &> /dev/null; then
    gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo -e "  ${GREEN}✓${NC} GPU found: $gpu_info"
    
    # Check CUDA in PyTorch
    python3 -c "import torch; print(f'  ✓ CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || \
        echo -e "  ${YELLOW}!${NC} CUDA not available in PyTorch"
else
    echo -e "  ${YELLOW}!${NC} No GPU detected - will use CPU mode"
fi
echo

# Step 8: Update config.yaml with reminder
echo -e "${GREEN}Step 8: Checking configuration...${NC}"
if [ -f "config.yaml" ]; then
    # Check if path needs updating
    if grep -q "/path/to" config.yaml; then
        echo -e "  ${YELLOW}!${NC} Remember to update model paths in config.yaml"
        echo -e "  ${YELLOW}!${NC} Edit with: vim config.yaml"
    else
        echo -e "  ${GREEN}✓${NC} config.yaml exists"
    fi
else
    echo -e "  ${RED}✗${NC} config.yaml missing!"
fi
echo

# Step 9: Make scripts executable
echo -e "${GREEN}Step 9: Making scripts executable...${NC}"
chmod +x quick_setup.sh 2>/dev/null || true
chmod +x model_manager.py 2>/dev/null || true
chmod +x bulk_tester.py 2>/dev/null || true
chmod +x llm_cli.py 2>/dev/null || true
echo -e "  ${GREEN}✓${NC} Scripts made executable"
echo

# Step 10: Validate setup
echo -e "${GREEN}Step 10: Validating setup...${NC}"

# Create simple validation
cat > validate_temp.py << 'EOF'
#!/usr/bin/env python3
import sys
import os

errors = []

# Check files
required_files = ['model_manager.py', 'bulk_tester.py', 'llm_cli.py', 
                  'config.yaml', 'Makefile', 'README.md']
for f in required_files:
    if not os.path.exists(f):
        errors.append(f"Missing file: {f}")

# Check imports
try:
    import torch
    import transformers
    import fastapi
    import pandas
    import yaml
    import click
    import rich
    print("  ✓ All core packages installed")
except ImportError as e:
    errors.append(f"Missing package: {e.name}")

# Check directories
for d in ['results', 'dataset_cache', 'sampled_datasets', 'logs']:
    if not os.path.exists(d):
        errors.append(f"Missing directory: {d}")

if errors:
    print("  ✗ Issues found:")
    for e in errors:
        print(f"    - {e}")
    sys.exit(1)
else:
    print("  ✓ All validations passed!")
    sys.exit(0)
EOF

if python3 validate_temp.py; then
    rm validate_temp.py
    echo
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}       ✓ SETUP COMPLETE!               ${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Edit config.yaml with your model paths:"
    echo "   ${YELLOW}vim config.yaml${NC}"
    echo
    echo "2. Run a quick test:"
    echo "   ${YELLOW}make quick${NC}"
    echo
    echo "3. View results:"
    echo "   ${YELLOW}make dashboard${NC}"
    echo
    echo -e "${GREEN}Happy testing! 🚀${NC}"
else
    rm validate_temp.py
    echo
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}       ✗ SETUP INCOMPLETE              ${NC}"
    echo -e "${RED}========================================${NC}"
    echo
    echo "Please fix the issues above and run again."
    exit 1
fi