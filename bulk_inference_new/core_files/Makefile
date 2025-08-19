# Simplified Makefile for Multi-LLM Testing Pipeline
# One-command operations for everything

# Variables
PYTHON = python3
PIP = pip3
ENV_NAME = llm-testing
CONDA_BASE = $(shell conda info --base 2>/dev/null || echo "")

# Color output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[1;33m
CYAN = \033[0;36m
NC = \033[0m # No Color

.PHONY: help install setup server test quick full clean status results dashboard

help: ## Show this help message
	@echo "$(CYAN)╔══════════════════════════════════════════════════════╗$(NC)"
	@echo "$(CYAN)║       Multi-LLM Testing Pipeline Commands           ║$(NC)"
	@echo "$(CYAN)╚══════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(GREEN)Quick Start:$(NC)"
	@echo "  make setup       # One-time setup"
	@echo "  make quick       # Run quick test on all models"
	@echo "  make dashboard   # View results in browser"
	@echo ""
	@echo "$(GREEN)Available Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Examples:$(NC)"
	@echo "  make test MODEL=llama-70b        # Test specific model"
	@echo "  make test SUITE=full_benchmark   # Run full benchmark"
	@echo "  make compare                      # Compare last two runs"

# ============= SETUP COMMANDS =============

install: ## Install Python dependencies
	@echo "$(GREEN)[Installing Dependencies]$(NC)"
	$(PIP) install -q torch transformers vllm fastapi uvicorn promptbench
	$(PIP) install -q pandas numpy tqdm pyyaml rich click
	$(PIP) install -q matplotlib seaborn plotly
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

setup: install ## Complete setup (install + create config)
	@echo "$(CYAN)[Setting Up Environment]$(NC)"
	@if [ ! -f config.yaml ]; then \
		$(PYTHON) -c "from llm_cli import *; cli(['config'])"; \
		echo "$(YELLOW)⚠ Edit config.yaml with your model paths$(NC)"; \
	else \
		echo "$(GREEN)✓ Config exists$(NC)"; \
	fi
	@mkdir -p results dataset_cache sampled_datasets
	@echo "$(GREEN)✓ Setup complete!$(NC)"
	@echo ""
	@echo "$(CYAN)Next steps:$(NC)"
	@echo "1. Edit config.yaml with your model paths"
	@echo "2. Run 'make quick' for a quick test"

# ============= SERVER COMMANDS =============

server: ## Start inference server
	@echo "$(GREEN)[Starting Server]$(NC)"
	@$(PYTHON) model_manager.py --server

server-bg: ## Start server in background
	@echo "$(GREEN)[Starting Server in Background]$(NC)"
	@nohup $(PYTHON) model_manager.py --server > server.log 2>&1 &
	@sleep 3
	@echo "$(GREEN)✓ Server started (PID: $$!)$(NC)"

stop: ## Stop inference server
	@echo "$(YELLOW)[Stopping Server]$(NC)"
	@pkill -f "model_manager.py" || true
	@echo "$(GREEN)✓ Server stopped$(NC)"

restart: stop server-bg ## Restart server

# ============= TESTING COMMANDS =============

quick: server-bg ## Run quick test suite (fastest)
	@echo "$(CYAN)[Running Quick Test]$(NC)"
	@$(PYTHON) bulk_tester.py --suite quick_test
	@make dashboard

test: server-bg ## Run tests (use MODEL=name or SUITE=name)
	@echo "$(CYAN)[Running Tests]$(NC)"
	@if [ -n "$(MODEL)" ]; then \
		$(PYTHON) bulk_tester.py --models $(MODEL) --suite $(or $(SUITE),quick_test); \
	elif [ -n "$(SUITE)" ]; then \
		$(PYTHON) bulk_tester.py --suite $(SUITE); \
	else \
		$(PYTHON) bulk_tester.py --suite quick_test; \
	fi

full: server-bg ## Run full benchmark suite
	@echo "$(CYAN)[Running Full Benchmark]$(NC)"
	@$(PYTHON) bulk_tester.py --suite full_benchmark
	@make dashboard

# ============= MODEL COMMANDS =============

models: ## List available models
	@$(PYTHON) -c "from llm_cli import *; controller = LLMTestingCLI(); controller.list_models()"

load: ## Load a model (use MODEL=name)
	@if [ -z "$(MODEL)" ]; then \
		echo "$(RED)Error: Specify MODEL=model_name$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)[Loading Model: $(MODEL)]$(NC)"
	@curl -X POST http://localhost:8000/load_model/$(MODEL)

# ============= RESULTS COMMANDS =============

results: ## Show latest results
	@$(PYTHON) -c "from llm_cli import *; controller = LLMTestingCLI(); controller.show_results()"

compare: ## Compare last two benchmark runs
	@$(PYTHON) bulk_tester.py --compare

dashboard: ## Open results dashboard in browser
	@echo "$(CYAN)[Opening Dashboard]$(NC)"
	@$(PYTHON) -c "import webbrowser; from pathlib import Path; \
		reports = list(Path('results').glob('report_*.html')); \
		webbrowser.open(f'file://{sorted(reports)[-1].absolute()}') if reports else print('No reports found')"

# ============= STATUS COMMANDS =============

status: ## Check system status
	@echo "$(CYAN)╔══════════════════════════════════════════════════════╗$(NC)"
	@echo "$(CYAN)║                    System Status                     ║$(NC)"
	@echo "$(CYAN)╚══════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(GREEN)Server Status:$(NC)"
	@curl -s http://localhost:8000/health 2>/dev/null && echo "  ✓ Running" || echo "  ✗ Not running"
	@echo ""
	@echo "$(GREEN)Current Model:$(NC)"
	@curl -s http://localhost:8000/models 2>/dev/null | $(PYTHON) -c "import sys, json; \
		data = json.load(sys.stdin) if sys.stdin else {}; \
		print(f\"  {data.get('current_model', 'None')}\")" 2>/dev/null || echo "  None"
	@echo ""
	@echo "$(GREEN)GPU Status:$(NC)"
	@nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null | \
		awk '{print "  GPU " NR-1 ": " $$0}' || echo "  No GPUs detected"

gpu: ## Monitor GPU usage
	@watch -n 1 nvidia-smi

# ============= UTILITY COMMANDS =============

datasets: ## List dataset suites
	@$(PYTHON) -c "from llm_cli import *; controller = LLMTestingCLI(); controller.list_datasets()"

clean: ## Clean results and cache
	@echo "$(YELLOW)[Cleaning]$(NC)"
	@rm -rf results/*.json results/*.csv results/*.html
	@rm -rf dataset_cache/*.json
	@rm -rf __pycache__ *.pyc
	@echo "$(GREEN)✓ Cleaned$(NC)"

clean-all: clean stop ## Full cleanup including stopping server
	@rm -rf results dataset_cache sampled_datasets
	@echo "$(GREEN)✓ Full cleanup complete$(NC)"

logs: ## View server logs
	@tail -f server.log 2>/dev/null || echo "No server logs found"

# ============= DOCKER COMMANDS =============

docker-build: ## Build Docker image
	@echo "$(CYAN)[Building Docker Image]$(NC)"
	@docker build -t llm-testing:latest .

docker-run: ## Run in Docker
	@docker run --gpus all -p 8000:8000 -v $(PWD)/models:/models llm-testing:latest

# ============= DEVELOPMENT COMMANDS =============

debug: ## Run with debug logging
	@LOG_LEVEL=DEBUG $(PYTHON) bulk_tester.py --suite quick_test

shell: ## Python shell with imports
	@$(PYTHON) -i -c "from model_manager import *; from bulk_tester import *; \
		print('Loaded: ModelManager, BulkTester')"

format: ## Format Python code
	@black *.py
	@echo "$(GREEN)✓ Code formatted$(NC)"

# ============= ONE-COMMAND WORKFLOWS =============

benchmark-all: setup server-bg ## Complete benchmark of all models
	@echo "$(CYAN)╔══════════════════════════════════════════════════════╗$(NC)"
	@echo "$(CYAN)║           Running Complete Benchmark Suite           ║$(NC)"
	@echo "$(CYAN)╚══════════════════════════════════════════════════════╝$(NC)"
	@$(PYTHON) bulk_tester.py --suite full_benchmark
	@make dashboard
	@make stop
	@echo "$(GREEN)✓ Benchmark complete! Check dashboard.$(NC)"

quick-compare: ## Quick test and compare with previous
	@make quick
	@sleep 2
	@make compare

# Default target
.DEFAULT_GOAL := help