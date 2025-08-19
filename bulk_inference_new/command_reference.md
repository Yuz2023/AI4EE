# 📝 Multi-LLM Testing Pipeline - Command Reference

## 🚀 Quick Start Commands

```bash
# Complete setup (run once)
./complete_setup.sh

# Quick test (5 minutes)
make quick

# View results
make dashboard
```

## 📊 Testing Commands

| Command | Description | Time |
|---------|-------------|------|
| `make quick` | Test all models with small dataset | ~5 min |
| `make full` | Complete benchmark suite | ~30 min |
| `make test MODEL=llama-70b` | Test specific model | ~5 min |
| `make test SUITE=full_benchmark` | Run specific test suite | Varies |
| `make benchmark-all` | Test everything and generate report | ~1 hour |

## 🖥️ Server Management

| Command | Description |
|---------|-------------|
| `make server` | Start server (foreground) |
| `make server-bg` | Start server (background) |
| `make stop` | Stop server |
| `make restart` | Restart server |
| `./start_server.sh` | Start with logging |
| `./stop_server.sh` | Stop gracefully |

## 📈 Results & Analysis

| Command | Description |
|---------|-------------|
| `make results` | Show latest results in terminal |
| `make dashboard` | Open HTML dashboard in browser |
| `make compare` | Compare last two runs |
| `ls results/` | List all result files |

## 🔍 Status & Information

| Command | Description |
|---------|-------------|
| `make status` | System status (server, GPU, model) |
| `make models` | List configured models |
| `make datasets` | List dataset suites |
| `make gpu` | Monitor GPU usage (live) |
| `make logs` | View server logs |

## 🛠️ Setup & Maintenance

| Command | Description |
|---------|-------------|
| `make setup` | Initial setup |
| `make install` | Install dependencies |
| `make clean` | Clean results and cache |
| `make clean-all` | Full cleanup |
| `python validate_setup.py` | Validate installation |
| `python final_check.py` | Comprehensive validation |

## 🐍 Python Scripts Direct Usage

### Model Manager
```bash
# List models
python model_manager.py --list

# Start server
python model_manager.py --server

# Custom port
python model_manager.py --server --port 8080
```

### Bulk Tester
```bash
# Run benchmark
python bulk_tester.py --suite quick_test

# Test specific models
python bulk_tester.py --models llama-70b llama-8b

# Compare results
python bulk_tester.py --compare
```

### CLI Interface
```bash
# Initialize
python llm_cli.py init

# Start server
python llm_cli.py server

# Show status
python llm_cli.py status

# List models
python llm_cli.py models

# Run tests
python llm_cli.py test --suite quick_test

# Generate text
python llm_cli.py generate "Hello, world"
```

## 🌐 API Endpoints

Once server is running on `http://localhost:8000`:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/models` | GET | List available models |
| `/load_model/{name}` | POST | Load specific model |
| `/generate` | POST | Generate text |
| `/batch_generate` | POST | Batch generation |

### API Examples

```bash
# Check health
curl http://localhost:8000/health

# List models
curl http://localhost:8000/models

# Load model
curl -X POST http://localhost:8000/load_model/llama-70b

# Generate text
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world"}'

# Batch generate
curl -X POST http://localhost:8000/batch_generate \
  -H "Content-Type: application/json" \
  -d '{"prompts": ["Test 1", "Test 2", "Test 3"]}'
```

## 🔧 Configuration

### Edit Configuration
```bash
vim config.yaml
```

### Configuration Structure
```yaml
models:
  model_name:
    path: "/path/to/model"
    tensor_parallel_size: 8
    
datasets:
  suite_name:
    - name: "dataset"
      samples: 100
      
server:
  port: 8000
  batch_size: 32
```

## 🐳 Docker Commands

| Command | Description |
|---------|-------------|
| `make docker-build` | Build Docker image |
| `make docker-run` | Run in Docker |
| `docker-compose up` | Start with compose |
| `docker-compose down` | Stop containers |

## 💡 Tips & Tricks

### Quick Testing Workflow
```bash
# 1. Start server in background
make server-bg

# 2. Run quick test
make quick

# 3. View results immediately
make dashboard

# 4. Stop server when done
make stop
```

### Production Workflow
```bash
# 1. Validate setup
python final_check.py

# 2. Run full benchmark
make benchmark-all

# 3. Results are auto-opened in browser
```

### Debug Mode
```bash
# Run with debug logging
LOG_LEVEL=DEBUG python bulk_tester.py --suite quick_test

# Interactive Python shell
make shell

# Check specific model
python -c "from model_manager import ModelManager; m = ModelManager(); print(m.list_models())"
```

## ⚡ Keyboard Shortcuts

When using `make gpu` or `make logs`:
- `Ctrl+C` - Exit monitoring
- `q` - Quit (in some views)

## 🆘 Troubleshooting Commands

```bash
# Check what's using port 8000
lsof -i :8000

# Kill all Python processes
pkill -f python

# Check GPU memory
nvidia-smi

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Check Python version
python --version

# List installed packages
pip list | grep -E "torch|transformers|vllm|fastapi"
```

## 📦 Environment Variables

Set these before running:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Select GPUs
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
```

Or use the `.env` file created by setup.

## 🎯 Common Workflows

### Add New Model
1. Edit `config.yaml`
2. Add model configuration
3. Run `make models` to verify
4. Test with `make test MODEL=new_model`

### Add New Dataset
1. Edit `config.yaml`
2. Add to dataset suite
3. Run `make datasets` to verify
4. Test with `make test SUITE=new_suite`

### Compare Models
1. `make test MODEL=model1`
2. `make test MODEL=model2`
3. `make compare`

### Generate Report
1. `make benchmark-all`
2. Report auto-opens
3. Find in `results/report_*.html`

---

**Pro Tip**: Use `make help` to see all available commands with descriptions!