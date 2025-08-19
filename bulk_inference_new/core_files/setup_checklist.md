# 📋 Multi-LLM Testing Pipeline Setup Checklist

## Step 1: File Preparation ✅

### Rename Files Correctly
```bash
# Core Python modules (use underscores)
mv model-manager.py model_manager.py
mv bulk-tester.py bulk_tester.py  
mv llm-cli.py llm_cli.py

# Configuration and docs
mv config-yaml.txt config.yaml
mv simplified-makefile.txt Makefile
mv readme-streamlined.md README.md

# Keep as-is
# quick_setup.sh - already correct
```

### Fix Character Encoding
```bash
# Run the encoding fix script
python fix_encoding.py
```

## Step 2: Directory Structure ✅

Create required directories:
```bash
mkdir -p results dataset_cache sampled_datasets logs
```

Final structure should be:
```
.
├── model_manager.py      ✅ Model loading and server
├── bulk_tester.py        ✅ Testing engine
├── llm_cli.py           ✅ CLI interface
├── config.yaml          ✅ Configuration
├── Makefile             ✅ Commands
├── quick_setup.sh       ✅ Setup script
├── README.md            ✅ Documentation
├── validate_setup.py    ✅ Validation script
├── fix_encoding.py      ✅ Encoding fixer
│
├── results/             📁 Test results
├── dataset_cache/       📁 Cached datasets
├── sampled_datasets/    📁 Sampled data
└── logs/               📁 Log files
```

## Step 3: Install Dependencies ✅

### Core Requirements
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
# Core ML
pip install torch transformers vllm accelerate

# Server
pip install fastapi uvicorn pydantic httpx

# Data & Testing  
pip install promptbench datasets pandas numpy

# Utilities
pip install pyyaml requests tqdm psutil

# CLI
pip install click rich

# Optional
pip install matplotlib seaborn
```

### Create requirements.txt
```txt
torch>=2.5.0
transformers>=4.46.0
vllm>=0.6.4
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
promptbench>=0.0.4
datasets>=2.19.0
pandas>=1.3.5
numpy>=1.26.0
pyyaml>=6.0
requests>=2.32.0
tqdm>=4.66.0
click>=8.0
rich>=13.0
pydantic>=2.0
```

## Step 4: Configure Models ✅

Edit `config.yaml`:
```yaml
models:
  llama-70b:
    path: "/actual/path/to/your/model"  # ← UPDATE THIS
    type: "llama"
    tensor_parallel_size: 8
    
  # Add more models as needed
```

## Step 5: Validate Setup ✅

```bash
# Run validation script
python validate_setup.py

# Should see all green checkmarks!
```

## Step 6: Test the System ✅

### Quick Test
```bash
# Start server and run quick test
make quick
```

### Manual Test
```bash
# Terminal 1: Start server
python model_manager.py --server

# Terminal 2: Test generation
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, test"}'

# Terminal 3: Run benchmark
python bulk_tester.py --suite quick_test
```

## Common Issues & Fixes 🔧

### Issue 1: Import Errors
```bash
# If you get "ModuleNotFoundError"
pip install [missing_module]
```

### Issue 2: CUDA/GPU Issues
```bash
# Check CUDA version
nvidia-smi

# Reinstall PyTorch for your CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Issue 3: vLLM Issues
```bash
# Install with CUDA support
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121
```

### Issue 4: Server Won't Start
```bash
# Check if port is in use
lsof -i :8000

# Kill existing process
pkill -f model_manager.py
```

### Issue 5: Out of Memory
```yaml
# In config.yaml, reduce batch size:
server:
  batch_size: 8  # Lower value
```

## Success Indicators ✅

You know everything is working when:

1. ✅ `python validate_setup.py` shows all green
2. ✅ `make status` shows server running
3. ✅ `make quick` completes without errors
4. ✅ Results appear in `results/` directory
5. ✅ HTML report opens in browser

## Quick Commands Reference 📝

```bash
# One-time setup
./quick_setup.sh

# Daily usage
make quick          # Quick test all models
make full           # Full benchmark
make dashboard      # View results
make compare        # Compare runs
make status         # Check status

# Specific tests
make test MODEL=llama-70b
make test SUITE=full_benchmark

# Management
make stop           # Stop server
make clean          # Clean results
make gpu            # Monitor GPU
```

## Support Checklist 🆘

Before asking for help, check:

- [ ] All files renamed correctly (underscores not hyphens)
- [ ] All dependencies installed (`pip list | grep vllm`)
- [ ] Config.yaml has valid model paths
- [ ] CUDA version matches PyTorch (`nvidia-smi` vs `torch.version.cuda`)
- [ ] Server is running (`curl http://localhost:8000/health`)
- [ ] No typos in file names or imports
- [ ] Python 3.8+ installed
- [ ] Sufficient GPU memory for models

## Ready to Test! 🎉

If all checks pass:
```bash
make benchmark-all
```

This will test all models on all datasets and generate a beautiful report!