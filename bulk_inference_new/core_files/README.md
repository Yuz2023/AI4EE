# 🚀 Streamlined Multi-LLM Testing Pipeline

A unified, easy-to-use testing framework for benchmarking multiple Large Language Models with just a few commands.

## ✨ What's New

- **One-command operations** - No more multi-step processes
- **Multi-model support** - Test different LLMs without code changes  
- **Centralized configuration** - Single YAML file controls everything
- **Automated benchmarking** - Run complete test suites automatically
- **Beautiful dashboards** - HTML reports with comparison tables
- **Smart model management** - Hot-swap models without restarting

## 📦 Quick Start (2 Minutes!)

```bash
# 1. Run the setup script
chmod +x quick_setup.sh
./quick_setup.sh

# 2. Edit config.yaml with your model paths
vim config.yaml

# 3. Run your first test!
make quick
```

That's it! Results will open automatically in your browser.

## 🎯 Key Features

### 1. Simplified Commands

| Command | What it does |
|---------|-------------|
| `make quick` | Run quick test on all models |
| `make full` | Run complete benchmark suite |
| `make dashboard` | View results in browser |
| `make compare` | Compare last two runs |
| `make status` | Check system status |

### 2. Easy Model Management

```yaml
# config.yaml - Add all your models here
models:
  llama-70b:
    path: "/path/to/llama-3.1-70b"
    tensor_parallel_size: 8
    
  llama-8b:
    path: "/path/to/llama-3.1-8b"
    tensor_parallel_size: 1
    
  mistral-7b:
    path: "/path/to/mistral-7b"
    tensor_parallel_size: 1
```

### 3. Flexible Dataset Suites

```yaml
datasets:
  quick_test:     # 5-minute test
    - name: "sst2"
      samples: 100
      
  full_benchmark: # 30-minute comprehensive test
    - name: "sst2"
      samples: 1000
    - name: "mmlu"
      samples: 500
    - name: "math"
      samples: 200
```

## 🔧 Installation

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA 12.1+ (optional but recommended)
- 32GB+ RAM
- 80GB+ GPU VRAM for 70B models

### Automated Setup

```bash
# Clone or download the scripts, then:
chmod +x quick_setup.sh
./quick_setup.sh
```

### Manual Setup

```bash
# Install dependencies
pip install torch transformers vllm fastapi uvicorn
pip install promptbench pandas pyyaml rich click tqdm

# Create directories
mkdir -p results dataset_cache sampled_datasets

# Create config
python llm_cli.py config
```

## 📊 Usage Examples

### Test a Specific Model

```bash
# Test just one model
make test MODEL=llama-70b

# Test with specific dataset suite
make test MODEL=llama-8b SUITE=full_benchmark
```

### Run Complete Benchmark

```bash
# Test all models on all datasets
make benchmark-all

# This will:
# 1. Start the server
# 2. Test each model
# 3. Generate reports
# 4. Open dashboard
# 5. Stop the server
```

### Compare Models

```bash
# Run tests on two models and compare
make test MODEL=llama-70b
make test MODEL=mistral-7b
make compare
```

### View Results

```bash
# Show latest results in terminal
make results

# Open interactive dashboard
make dashboard

# Compare last two runs
make compare
```

## 📁 Project Structure

```
.
├── config.yaml              # ⭐ Main configuration file
├── model_manager.py         # Model loading and server
├── bulk_tester.py          # Automated testing engine
├── llm_cli.py              # Command-line interface
├── Makefile                # One-command operations
├── quick_setup.sh          # Automated setup script
│
├── results/                # Test results and reports
│   ├── benchmark_summary_*.csv
│   └── report_*.html
│
├── dataset_cache/          # Cached datasets
└── sampled_datasets/       # Sampled data for quick tests
```

## 🎛️ Configuration Guide

### Basic Configuration

```yaml
# config.yaml
models:
  model_name:
    path: "/absolute/path/to/model"
    tensor_parallel_size: 8  # Number of GPUs
    default_sampling:
      temperature: 0.1
      top_p: 0.9
      max_tokens: 512

datasets:
  suite_name:
    - name: "dataset_name"
      samples: 100
      metrics: ["accuracy", "latency"]

server:
  port: 8000
  batch_size: 32
```

### Advanced Options

```yaml
# Performance tuning
performance:
  pytorch_cuda_alloc_conf: "max_split_size_mb:512"
  num_workers: 4

# Custom dataset suites
datasets:
  my_custom_suite:
    - name: "sst2"
      samples: 50
    - name: "mmlu"
      subset: "physics"
      samples: 100
```

## 📈 Understanding Results

### Terminal Output

```
BENCHMARK RESULTS SUMMARY
====================================================================
     Model    Dataset  Accuracy  Valid Format  Latency (s)
  llama-70b      sst2     92.3%        100.0%        0.234
  llama-70b      mmlu     78.5%         98.5%        0.312
  llama-8b       sst2     88.1%         99.8%        0.089
  llama-8b       mmlu     65.2%         97.2%        0.125
```

### HTML Dashboard

The dashboard shows:
- **Accuracy comparison table** - Side-by-side model performance
- **Valid format rates** - How often models produce correct output format
- **Latency metrics** - Response time per query
- **Detailed results** - Full test data for analysis

## 🐛 Troubleshooting

### Server Won't Start

```bash
# Check if port is in use
lsof -i :8000

# Kill existing process
make stop

# Start fresh
make server
```

### Out of Memory

```bash
# Reduce batch size in config.yaml
server:
  batch_size: 8  # Smaller batches

# Or test smaller models first
make test MODEL=llama-8b
```

### CUDA Errors

```bash
# Check CUDA version
nvidia-smi

# Reinstall PyTorch for your CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 🚀 Performance Tips

1. **Start with quick tests** - Use `quick_test` suite to verify setup
2. **Cache datasets** - Enable `use_sampled_data: true` for faster testing
3. **Adjust batch sizes** - Larger batches = faster but more memory
4. **Use tensor parallelism** - Distribute large models across GPUs
5. **Monitor GPU usage** - Run `make gpu` in separate terminal

## 📝 Adding New Models

1. Add to `config.yaml`:
```yaml
models:
  my_new_model:
    path: "/path/to/model"
    type: "llama"  # or "mistral", etc.
    tensor_parallel_size: 1
```

2. Test it:
```bash
make test MODEL=my_new_model
```

## 📊 Adding New Datasets

1. Add to `config.yaml`:
```yaml
datasets:
  my_suite:
    - name: "new_dataset"
      samples: 100
      metrics: ["accuracy"]
```

2. Run benchmark:
```bash
make test SUITE=my_suite
```

## 🤝 Contributing

Feel free to extend the pipeline with:
- New dataset processors
- Additional metrics
- Custom visualizations
- Model-specific optimizations

## 📄 License

MIT License - Use freely for research and development.

## 🆘 Support

- Check `make help` for all available commands
- Review logs in `results/` directory
- Ensure model paths are absolute paths
- Verify CUDA compatibility

---

**Happy Testing! 🎉**

For questions or issues, check the troubleshooting section or review the server logs.