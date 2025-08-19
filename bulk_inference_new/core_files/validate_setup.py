#!/usr/bin/env python3
"""
Validation script to ensure all components are properly set up
for the Multi-LLM Testing Pipeline
"""

import os
import sys
import importlib.util
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_status(status: str, message: str):
    """Print colored status messages"""
    if status == "success":
        print(f"{Colors.GREEN}✓{Colors.END} {message}")
    elif status == "error":
        print(f"{Colors.RED}✗{Colors.END} {message}")
    elif status == "warning":
        print(f"{Colors.YELLOW}!{Colors.END} {message}")
    elif status == "info":
        print(f"{Colors.BLUE}ℹ{Colors.END} {message}")

def check_file_exists(filepath: str) -> bool:
    """Check if a file exists"""
    return Path(filepath).exists()

def check_python_import(module_name: str) -> bool:
    """Check if a Python module can be imported"""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

def check_python_syntax(filepath: str) -> Tuple[bool, str]:
    """Check Python file syntax"""
    try:
        with open(filepath, 'r') as f:
            compile(f.read(), filepath, 'exec')
        return True, ""
    except SyntaxError as e:
        return False, str(e)

def validate_files() -> Dict[str, bool]:
    """Validate all required files exist"""
    print("\n" + "="*50)
    print("VALIDATING FILE STRUCTURE")
    print("="*50)
    
    required_files = {
        'model_manager.py': 'Model management and server',
        'bulk_tester.py': 'Automated testing engine',
        'llm_cli.py': 'Command-line interface',
        'config.yaml': 'Configuration file',
        'Makefile': 'Make commands',
        'quick_setup.sh': 'Setup script',
        'README.md': 'Documentation'
    }
    
    results = {}
    for filename, description in required_files.items():
        if check_file_exists(filename):
            print_status("success", f"{filename} - {description}")
            results[filename] = True
        else:
            print_status("error", f"{filename} - {description} [MISSING]")
            results[filename] = False
    
    return results

def validate_python_syntax() -> bool:
    """Validate Python files have correct syntax"""
    print("\n" + "="*50)
    print("VALIDATING PYTHON SYNTAX")
    print("="*50)
    
    python_files = ['model_manager.py', 'bulk_tester.py', 'llm_cli.py']
    all_valid = True
    
    for filepath in python_files:
        if check_file_exists(filepath):
            valid, error = check_python_syntax(filepath)
            if valid:
                print_status("success", f"{filepath} - Syntax OK")
            else:
                print_status("error", f"{filepath} - Syntax Error: {error}")
                all_valid = False
        else:
            print_status("warning", f"{filepath} - File not found")
            all_valid = False
    
    return all_valid

def validate_dependencies() -> Dict[str, bool]:
    """Check if all required Python packages are installed"""
    print("\n" + "="*50)
    print("VALIDATING DEPENDENCIES")
    print("="*50)
    
    required_packages = {
        # Core ML
        'torch': 'PyTorch - Deep learning framework',
        'transformers': 'Transformers - Hugging Face models',
        'vllm': 'vLLM - Fast LLM inference',
        
        # Server
        'fastapi': 'FastAPI - Web framework',
        'uvicorn': 'Uvicorn - ASGI server',
        'pydantic': 'Pydantic - Data validation',
        
        # Data & Testing
        'promptbench': 'PromptBench - Dataset loader',
        'pandas': 'Pandas - Data analysis',
        'numpy': 'NumPy - Numerical computing',
        
        # Utilities
        'yaml': 'PyYAML - YAML parser',
        'requests': 'Requests - HTTP library',
        'tqdm': 'tqdm - Progress bars',
        
        # CLI
        'click': 'Click - CLI framework',
        'rich': 'Rich - Terminal formatting'
    }
    
    results = {}
    missing = []
    
    for package, description in required_packages.items():
        if check_python_import(package):
            print_status("success", f"{package} - {description}")
            results[package] = True
        else:
            print_status("error", f"{package} - {description} [NOT INSTALLED]")
            results[package] = False
            missing.append(package)
    
    if missing:
        print_status("info", f"\nInstall missing packages with:")
        print(f"  pip install {' '.join(missing)}")
    
    return results

def validate_config() -> bool:
    """Validate config.yaml structure"""
    print("\n" + "="*50)
    print("VALIDATING CONFIGURATION")
    print("="*50)
    
    if not check_file_exists('config.yaml'):
        print_status("error", "config.yaml not found")
        return False
    
    try:
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['models', 'datasets', 'server', 'inference']
        for section in required_sections:
            if section in config:
                print_status("success", f"Config section '{section}' found")
            else:
                print_status("error", f"Config section '{section}' missing")
                return False
        
        # Check for at least one model
        if config.get('models'):
            model_count = len(config['models'])
            print_status("info", f"Found {model_count} model(s) configured")
            
            # Check model paths
            for name, model_config in config['models'].items():
                path = model_config.get('path', '')
                if Path(path).exists():
                    print_status("success", f"Model '{name}' path exists: {path}")
                else:
                    print_status("warning", f"Model '{name}' path not found: {path}")
        else:
            print_status("error", "No models configured")
            return False
        
        return True
        
    except Exception as e:
        print_status("error", f"Error reading config.yaml: {e}")
        return False

def validate_directories() -> bool:
    """Check and create required directories"""
    print("\n" + "="*50)
    print("VALIDATING DIRECTORIES")
    print("="*50)
    
    required_dirs = ['results', 'dataset_cache', 'sampled_datasets', 'logs']
    
    for dirname in required_dirs:
        if Path(dirname).exists():
            print_status("success", f"Directory '{dirname}' exists")
        else:
            try:
                Path(dirname).mkdir(parents=True, exist_ok=True)
                print_status("success", f"Created directory '{dirname}'")
            except Exception as e:
                print_status("error", f"Failed to create '{dirname}': {e}")
                return False
    
    return True

def check_gpu() -> bool:
    """Check GPU availability"""
    print("\n" + "="*50)
    print("CHECKING GPU/CUDA")
    print("="*50)
    
    # Check nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                              capture_output=True, text=True, check=True)
        gpus = result.stdout.strip().split('\n')
        print_status("success", f"Found {len(gpus)} GPU(s):")
        for gpu in gpus:
            print(f"  - {gpu}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_status("warning", "No NVIDIA GPU detected (CPU mode will be used)")
        return False
    
    # Check CUDA in PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            print_status("success", f"CUDA available in PyTorch: {torch.version.cuda}")
            print_status("info", f"GPU count: {torch.cuda.device_count()}")
            return True
        else:
            print_status("warning", "CUDA not available in PyTorch")
            return False
    except ImportError:
        print_status("error", "PyTorch not installed")
        return False

def test_imports() -> bool:
    """Test critical imports from our modules"""
    print("\n" + "="*50)
    print("TESTING MODULE IMPORTS")
    print("="*50)
    
    all_valid = True
    
    # Test model_manager imports
    if check_file_exists('model_manager.py'):
        try:
            from model_manager import ModelManager, InferenceServer
            print_status("success", "model_manager.py imports OK")
        except Exception as e:
            print_status("error", f"model_manager.py import error: {e}")
            all_valid = False
    
    # Test bulk_tester imports
    if check_file_exists('bulk_tester.py'):
        try:
            from bulk_tester import BulkTester
            print_status("success", "bulk_tester.py imports OK")
        except Exception as e:
            print_status("error", f"bulk_tester.py import error: {e}")
            all_valid = False
    
    # Test llm_cli imports
    if check_file_exists('llm_cli.py'):
        try:
            from llm_cli import LLMTestingCLI
            print_status("success", "llm_cli.py imports OK")
        except Exception as e:
            print_status("error", f"llm_cli.py import error: {e}")
            all_valid = False
    
    return all_valid

def generate_summary(results: Dict) -> None:
    """Generate validation summary"""
    print("\n" + "="*50)
    print("VALIDATION SUMMARY")
    print("="*50)
    
    total_checks = sum(len(v) if isinstance(v, dict) else 1 for v in results.values())
    passed_checks = sum(
        sum(v.values()) if isinstance(v, dict) else (1 if v else 0) 
        for v in results.values()
    )
    
    percentage = (passed_checks / total_checks * 100) if total_checks > 0 else 0
    
    if percentage == 100:
        print(f"{Colors.GREEN}✓ ALL CHECKS PASSED ({passed_checks}/{total_checks}){Colors.END}")
        print("\nYour Multi-LLM Testing Pipeline is ready to use!")
        print("\nNext steps:")
        print("  1. Edit config.yaml with your model paths")
        print("  2. Run: make quick")
        print("  3. View results: make dashboard")
    elif percentage >= 80:
        print(f"{Colors.YELLOW}⚠ MOSTLY READY ({passed_checks}/{total_checks} checks passed - {percentage:.1f}%){Colors.END}")
        print("\nFix the remaining issues above, then you're good to go!")
    else:
        print(f"{Colors.RED}✗ NOT READY ({passed_checks}/{total_checks} checks passed - {percentage:.1f}%){Colors.END}")
        print("\nPlease fix the issues above before proceeding.")
        print("Run: ./quick_setup.sh for automated setup")

def main():
    """Main validation routine"""
    print(f"\n{Colors.BLUE}{'='*50}{Colors.END}")
    print(f"{Colors.BLUE}Multi-LLM Testing Pipeline Validation{Colors.END}")
    print(f"{Colors.BLUE}{'='*50}{Colors.END}")
    
    results = {}
    
    # Run all validations
    results['files'] = validate_files()
    results['syntax'] = validate_python_syntax()
    results['dependencies'] = validate_dependencies()
    results['config'] = validate_config()
    results['directories'] = validate_directories()
    results['gpu'] = check_gpu()
    results['imports'] = test_imports()
    
    # Generate summary
    generate_summary(results)
    
    # Return exit code
    all_critical_passed = (
        all(results['files'].values()) and
        results['syntax'] and
        results['config'] and
        results['directories']
    )
    
    sys.exit(0 if all_critical_passed else 1)

if __name__ == "__main__":
    main()