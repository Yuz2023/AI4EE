#!/usr/bin/env python3
"""
Final validation script to ensure the Multi-LLM Testing Pipeline is ready
Run this after setup to verify everything is in place
"""

import os
import sys
import json
import yaml
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.CYAN}{'='*60}{Colors.END}")
    print(f"{Colors.CYAN}{text:^60}{Colors.END}")
    print(f"{Colors.CYAN}{'='*60}{Colors.END}")

def print_status(status: str, message: str, detail: str = ""):
    """Print formatted status message"""
    symbols = {
        "success": f"{Colors.GREEN}✓{Colors.END}",
        "error": f"{Colors.FAIL}✗{Colors.END}",
        "warning": f"{Colors.WARNING}⚠{Colors.END}",
        "info": f"{Colors.BLUE}ℹ{Colors.END}"
    }
    
    symbol = symbols.get(status, "•")
    if detail:
        print(f"  {symbol} {message}: {Colors.BOLD}{detail}{Colors.END}")
    else:
        print(f"  {symbol} {message}")

def check_required_files() -> Tuple[bool, List[str]]:
    """Check all required files exist"""
    print_header("CHECKING REQUIRED FILES")
    
    required_files = {
        # Core modules
        "model_manager.py": "Model management and server",
        "bulk_tester.py": "Automated testing engine",
        "llm_cli.py": "Command-line interface",
        
        # Configuration
        "config.yaml": "Main configuration",
        "requirements.txt": "Python dependencies",
        "Makefile": "Command shortcuts",
        
        # Documentation
        "README.md": "Documentation",
    }
    
    missing_files = []
    all_present = True
    
    for filename, description in required_files.items():
        if Path(filename).exists():
            size = Path(filename).stat().st_size
            print_status("success", f"{filename:<25} {description}", f"{size:,} bytes")
        else:
            print_status("error", f"{filename:<25} {description}", "MISSING")
            missing_files.append(filename)
            all_present = False
    
    return all_present, missing_files

def check_directories() -> bool:
    """Check and create required directories"""
    print_header("CHECKING DIRECTORIES")
    
    required_dirs = [
        "results",
        "dataset_cache", 
        "sampled_datasets",
        "logs"
    ]
    
    all_present = True
    for dirname in required_dirs:
        path = Path(dirname)
        if path.exists():
            num_files = len(list(path.glob("*")))
            print_status("success", f"{dirname:<20}", f"{num_files} files")
        else:
            try:
                path.mkdir(parents=True, exist_ok=True)
                print_status("warning", f"{dirname:<20}", "Created")
            except Exception as e:
                print_status("error", f"{dirname:<20}", f"Failed: {e}")
                all_present = False
    
    return all_present

def check_python_modules() -> Tuple[bool, List[str]]:
    """Check if Python modules can be imported"""
    print_header("CHECKING PYTHON MODULES")
    
    modules_to_check = [
        ("model_manager", "ModelManager", "Model management"),
        ("bulk_tester", "BulkTester", "Testing engine"),
        ("llm_cli", "LLMTestingCLI", "CLI interface")
    ]
    
    all_valid = True
    errors = []
    
    for module_name, class_name, description in modules_to_check:
        try:
            module = __import__(module_name)
            if hasattr(module, class_name):
                print_status("success", f"{module_name:<20} {description}")
            else:
                print_status("warning", f"{module_name:<20} Missing class: {class_name}")
                errors.append(f"{module_name}: Missing {class_name}")
        except SyntaxError as e:
            print_status("error", f"{module_name:<20} Syntax error: {e}")
            errors.append(f"{module_name}: Syntax error")
            all_valid = False
        except Exception as e:
            print_status("error", f"{module_name:<20} Import error: {e}")
            errors.append(f"{module_name}: Import failed")
            all_valid = False
    
    return all_valid, errors

def check_dependencies() -> Tuple[bool, List[str]]:
    """Check if required Python packages are installed"""
    print_header("CHECKING DEPENDENCIES")
    
    required_packages = {
        "torch": "PyTorch",
        "transformers": "Hugging Face Transformers",
        "vllm": "vLLM inference engine",
        "fastapi": "FastAPI web framework",
        "uvicorn": "ASGI server",
        "promptbench": "Dataset loader",
        "pandas": "Data processing",
        "numpy": "Numerical computing",
        "yaml": "YAML parser",
        "requests": "HTTP library",
        "tqdm": "Progress bars",
        "click": "CLI framework",
        "rich": "Terminal formatting"
    }
    
    missing_packages = []
    all_installed = True
    
    for package, description in required_packages.items():
        try:
            __import__(package)
            # Get version if possible
            try:
                module = __import__(package)
                version = getattr(module, "__version__", "unknown")
                print_status("success", f"{package:<15} {description}", version)
            except:
                print_status("success", f"{package:<15} {description}")
        except ImportError:
            print_status("error", f"{package:<15} {description}", "NOT INSTALLED")
            missing_packages.append(package)
            all_installed = False
    
    return all_installed, missing_packages

def check_configuration() -> Tuple[bool, Dict]:
    """Validate config.yaml structure and content"""
    print_header("CHECKING CONFIGURATION")
    
    if not Path("config.yaml").exists():
        print_status("error", "config.yaml not found")
        return False, {}
    
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        issues = []
        
        # Check required sections
        required_sections = ["models", "datasets", "server", "inference"]
        for section in required_sections:
            if section in config:
                items = len(config[section]) if isinstance(config[section], dict) else "configured"
                print_status("success", f"Section '{section}'", f"{items}")
            else:
                print_status("error", f"Section '{section}'", "MISSING")
                issues.append(f"Missing section: {section}")
        
        # Check model configurations
        if "models" in config:
            print_status("info", "Model Configurations:")
            for model_name, model_config in config["models"].items():
                path = model_config.get("path", "")
                if Path(path).exists():
                    print_status("success", f"  {model_name}", path[:50] + "...")
                else:
                    print_status("warning", f"  {model_name}", f"Path not found: {path[:50]}...")
                    issues.append(f"Model path not found: {model_name}")
        
        return len(issues) == 0, config
        
    except yaml.YAMLError as e:
        print_status("error", "Invalid YAML", str(e))
        return False, {}
    except Exception as e:
        print_status("error", "Error reading config", str(e))
        return False, {}

def check_gpu() -> bool:
    """Check GPU availability"""
    print_header("CHECKING GPU/CUDA")
    
    has_gpu = False
    
    # Check nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True
        )
        gpus = result.stdout.strip().split('\n')
        print_status("success", f"Found {len(gpus)} GPU(s):")
        for gpu in gpus:
            print(f"    {gpu}")
        has_gpu = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_status("warning", "No NVIDIA GPU detected", "CPU mode will be used")
    
    # Check CUDA in PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            print_status("success", "CUDA available in PyTorch", torch.version.cuda)
            print_status("info", "CUDA device count", str(torch.cuda.device_count()))
        else:
            print_status("warning", "CUDA not available in PyTorch")
    except ImportError:
        print_status("error", "PyTorch not installed")
    
    return has_gpu

def check_server() -> bool:
    """Check if server can be reached"""
    print_header("CHECKING SERVER")
    
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            print_status("success", "Server is running")
            print_status("info", "Current model", data.get("current_model", "None"))
            return True
        else:
            print_status("warning", "Server returned", f"Status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_status("info", "Server not running", "Start with: make server")
        return False
    except Exception as e:
        print_status("error", "Error checking server", str(e))
        return False

def generate_fix_commands(missing_files: List[str], missing_packages: List[str]):
    """Generate commands to fix issues"""
    print_header("FIX COMMANDS")
    
    if missing_packages:
        print_status("info", "Install missing packages:")
        print(f"    pip install {' '.join(missing_packages)}")
        print()
    
    if missing_files:
        print_status("info", "Missing files detected. Solutions:")
        
        if "config.yaml" in missing_files:
            print("    # Create config.yaml:")
            print("    python llm_cli.py config")
        
        if "requirements.txt" in missing_files:
            print("    # Create requirements.txt:")
            print("    echo 'torch>=2.5.0' > requirements.txt")
            print("    echo 'transformers>=4.46.0' >> requirements.txt")
            print("    echo 'vllm>=0.6.4' >> requirements.txt")
        
        if "Makefile" in missing_files:
            print("    # Get Makefile from the streamlined version")
        
        print()

def main():
    """Main validation routine"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'Multi-LLM Testing Pipeline - Final Validation':^60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
    
    # Track overall status
    all_checks_passed = True
    critical_issues = []
    warnings = []
    
    # 1. Check files
    files_ok, missing_files = check_required_files()
    if not files_ok:
        all_checks_passed = False
        critical_issues.append("Missing required files")
    
    # 2. Check directories
    dirs_ok = check_directories()
    
    # 3. Check Python modules
    modules_ok, module_errors = check_python_modules()
    if not modules_ok:
        all_checks_passed = False
        critical_issues.append("Python module errors")
    
    # 4. Check dependencies
    deps_ok, missing_packages = check_dependencies()
    if not deps_ok:
        warnings.append("Missing Python packages")
    
    # 5. Check configuration
    config_ok, config = check_configuration()
    if not config_ok:
        warnings.append("Configuration issues")
    
    # 6. Check GPU
    has_gpu = check_gpu()
    if not has_gpu:
        warnings.append("No GPU detected (CPU mode)")
    
    # 7. Check server
    server_running = check_server()
    
    # Generate fix commands if needed
    if missing_files or missing_packages:
        generate_fix_commands(missing_files, missing_packages)
    
    # Final summary
    print_header("VALIDATION SUMMARY")
    
    if all_checks_passed and not warnings:
        print(f"{Colors.GREEN}{Colors.BOLD}")
        print("  ✓ ALL CHECKS PASSED!")
        print("  Your Multi-LLM Testing Pipeline is ready to use!")
        print(f"{Colors.END}")
        print("\n  Next steps:")
        print("    1. Edit config.yaml with your model paths")
        print("    2. Run: make quick")
        print("    3. View results: make dashboard")
        return 0
    
    elif all_checks_passed:
        print(f"{Colors.WARNING}{Colors.BOLD}")
        print("  ⚠ READY WITH WARNINGS")
        print(f"{Colors.END}")
        for warning in warnings:
            print(f"    - {warning}")
        print("\n  The system should work, but address warnings for best performance.")
        return 0
    
    else:
        print(f"{Colors.FAIL}{Colors.BOLD}")
        print("  ✗ CRITICAL ISSUES FOUND")
        print(f"{Colors.END}")
        for issue in critical_issues:
            print(f"    - {issue}")
        print("\n  Fix the critical issues before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())