#!/usr/bin/env python3
"""
Unified CLI for LLM Testing Pipeline
One command to rule them all!
"""

import click
import subprocess
import yaml
import time
import requests
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich import print as rprint
import json

console = Console()

class LLMTestingCLI:
    """Main CLI controller"""
    
    def __init__(self):
        self.config_path = Path("config.yaml")
        self.config = self._load_config()
        
    def _load_config(self):
        """Load configuration"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def check_server_status(self):
        """Check if server is running"""
        try:
            port = self.config.get('server', {}).get('port', 8000)
            response = requests.get(f"http://localhost:{port}/health")
            return response.status_code == 200
        except:
            return False
    
    def start_server(self, background=True):
        """Start the inference server"""
        if self.check_server_status():
            console.print("[yellow]Server already running[/yellow]")
            return True
        
        console.print("[cyan]Starting inference server...[/cyan]")
        
        if background:
            # Start in background
            subprocess.Popen(
                ["python", "model_manager.py", "--server"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Wait for server to start
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Waiting for server to start...", total=None)
                
                for _ in range(30):  # 30 second timeout
                    if self.check_server_status():
                        progress.stop()
                        console.print("[green]✓ Server started successfully[/green]")
                        return True
                    time.sleep(1)
            
            console.print("[red]✗ Server failed to start[/red]")
            return False
        else:
            # Start in foreground
            subprocess.run(["python", "model_manager.py", "--server"])
            return True
    
    def stop_server(self):
        """Stop the inference server"""
        # Find and kill the process
        try:
            subprocess.run(["pkill", "-f", "model_manager.py"], check=False)
            console.print("[green]✓ Server stopped[/green]")
        except Exception as e:
            console.print(f"[red]Error stopping server: {e}[/red]")
    
    def list_models(self):
        """List available models"""
        table = Table(title="Available Models")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Path", style="green")
        table.add_column("GPUs", style="yellow")
        
        for name, config in self.config.get('models', {}).items():
            table.add_row(
                name,
                config.get('type', 'unknown'),
                config.get('path', 'N/A')[:50] + "...",
                str(config.get('tensor_parallel_size', 1))
            )
        
        console.print(table)
    
    def list_datasets(self):
        """List dataset suites"""
        for suite_name, datasets in self.config.get('datasets', {}).items():
            table = Table(title=f"Dataset Suite: {suite_name}")
            table.add_column("Dataset", style="cyan")
            table.add_column("Samples", style="yellow")
            table.add_column("Subset", style="magenta")
            table.add_column("Metrics", style="green")
            
            for dataset in datasets:
                table.add_row(
                    dataset['name'],
                    str(dataset.get('samples', 'all')),
                    dataset.get('subset', 'default'),
                    ', '.join(dataset.get('metrics', ['accuracy']))
                )
            
            console.print(table)
            console.print()
    
    def show_results(self, latest_only=True):
        """Show benchmark results"""
        results_dir = Path(self.config.get('inference', {}).get('output_dir', 'results'))
        
        if latest_only:
            # Find latest summary
            summaries = list(results_dir.glob("benchmark_summary_*.csv"))
            if not summaries:
                console.print("[yellow]No results found[/yellow]")
                return
            
            latest = sorted(summaries)[-1]
            df = pd.read_csv(latest)
            
            # Create nice table
            table = Table(title=f"Latest Results: {latest.stem}")
            table.add_column("Model", style="cyan")
            table.add_column("Dataset", style="magenta")
            table.add_column("Accuracy", style="green")
            table.add_column("Valid Format", style="yellow")
            table.add_column("Latency (s)", style="blue")
            
            for _, row in df.iterrows():
                table.add_row(
                    row['model'],
                    row['dataset'],
                    f"{row['accuracy']:.1f}%",
                    f"{row['valid_format_rate']:.1f}%",
                    f"{row['avg_latency']:.3f}"
                )
            
            console.print(table)
        else:
            # Show all results
            summaries = list(results_dir.glob("benchmark_summary_*.csv"))
            console.print(f"Found {len(summaries)} benchmark results:")
            for summary in sorted(summaries):
                console.print(f"  - {summary.name}")

@click.group()
def cli():
    """LLM Testing Pipeline CLI"""
    pass

@cli.command()
def init():
    """Initialize the testing environment"""
    controller = LLMTestingCLI()
    
    console.print(Panel.fit("🚀 LLM Testing Environment Initialization", style="bold blue"))
    
    # Check conda environment
    console.print("\n[cyan]Checking environment...[/cyan]")
    
    # Check for required files
    required_files = ['config.yaml', 'model_manager.py', 'bulk_tester.py']
    missing = [f for f in required_files if not Path(f).exists()]
    
    if missing:
        console.print(f"[red]Missing files: {', '.join(missing)}[/red]")
        console.print("[yellow]Please ensure all required files are present[/yellow]")
        return
    
    # Install dependencies if needed
    with console.status("Installing dependencies..."):
        subprocess.run(
            ["pip", "install", "-q", "pyyaml", "pandas", "rich", "click", "tqdm", "promptbench"],
            check=False
        )
    
    console.print("[green]✓ Environment ready![/green]")

@cli.command()
@click.option('--background/--foreground', default=True, help='Run server in background')
def server(background):
    """Start the inference server"""
    controller = LLMTestingCLI()
    controller.start_server(background)

@cli.command()
def stop():
    """Stop the inference server"""
    controller = LLMTestingCLI()
    controller.stop_server()

@cli.command()
def status():
    """Check system status"""
    controller = LLMTestingCLI()
    
    console.print(Panel.fit("📊 System Status", style="bold blue"))
    
    # Server status
    if controller.check_server_status():
        console.print("[green]✓ Server: Running[/green]")
        
        # Get current model
        try:
            port = controller.config.get('server', {}).get('port', 8000)
            response = requests.get(f"http://localhost:{port}/models")
            if response.status_code == 200:
                data = response.json()
                console.print(f"[cyan]Current Model: {data.get('current_model', 'None')}[/cyan]")
        except:
            pass
    else:
        console.print("[red]✗ Server: Not running[/red]")
    
    # GPU status
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            console.print("\n[bold]GPU Status:[/bold]")
            for line in result.stdout.strip().split('\n'):
                console.print(f"  {line}")
    except:
        console.print("[yellow]GPU information not available[/yellow]")

@cli.command()
def models():
    """List available models"""
    controller = LLMTestingCLI()
    controller.list_models()

@cli.command()
def datasets():
    """List dataset suites"""
    controller = LLMTestingCLI()
    controller.list_datasets()

@cli.command()
@click.argument('model_name')
def load(model_name):
    """Load a specific model"""
    controller = LLMTestingCLI()
    
    if not controller.check_server_status():
        console.print("[red]Server not running. Start it with: llm server[/red]")
        return
    
    port = controller.config.get('server', {}).get('port', 8000)
    
    with console.status(f"Loading model {model_name}..."):
        response = requests.post(f"http://localhost:{port}/load_model/{model_name}")
        
    if response.status_code == 200:
        console.print(f"[green]✓ Loaded model: {model_name}[/green]")
    else:
        console.print(f"[red]✗ Failed to load model: {response.text}[/red]")

@cli.command()
@click.option('--models', '-m', multiple=True, help='Models to test (can specify multiple)')
@click.option('--suite', '-s', default='quick_test', 
              type=click.Choice(['quick_test', 'full_benchmark', 'translation_test']),
              help='Dataset suite to run')
@click.option('--parallel', '-p', is_flag=True, help='Run tests in parallel')
def test(models, suite, parallel):
    """Run benchmark tests"""
    controller = LLMTestingCLI()
    
    console.print(Panel.fit(f"🧪 Running Benchmark: {suite}", style="bold blue"))
    
    # Ensure server is running
    if not controller.check_server_status():
        console.print("[yellow]Starting server...[/yellow]")
        if not controller.start_server():
            console.print("[red]Failed to start server[/red]")
            return
    
    # Prepare models list
    if not models:
        models = list(controller.config.get('models', {}).keys())
    
    console.print(f"[cyan]Testing models: {', '.join(models)}[/cyan]")
    console.print(f"[cyan]Dataset suite: {suite}[/cyan]\n")
    
    # Run the benchmark
    subprocess.run([
        "python", "bulk_tester.py",
        "--suite", suite,
        "--models"] + list(models)
    )

@cli.command()
@click.option('--latest/--all', default=True, help='Show latest or all results')
@click.option('--compare', is_flag=True, help='Compare last two runs')
def results(latest, compare):
    """Show benchmark results"""
    controller = LLMTestingCLI()
    
    if compare:
        subprocess.run(["python", "bulk_tester.py", "--compare"])
    else:
        controller.show_results(latest_only=latest)

@cli.command()
@click.argument('prompt')
@click.option('--model', '-m', help='Model to use (uses current if not specified)')
def generate(prompt, model):
    """Quick text generation"""
    controller = LLMTestingCLI()
    
    if not controller.check_server_status():
        console.print("[red]Server not running. Start it with: llm server[/red]")
        return
    
    port = controller.config.get('server', {}).get('port', 8000)
    
    # Load model if specified
    if model:
        response = requests.post(f"http://localhost:{port}/load_model/{model}")
        if response.status_code != 200:
            console.print(f"[red]Failed to load model {model}[/red]")
            return
    
    # Generate text
    with console.status("Generating..."):
        response = requests.post(
            f"http://localhost:{port}/generate",
            json={"prompt": prompt}
        )
    
    if response.status_code == 200:
        data = response.json()
        console.print(Panel(
            data['output'],
            title=f"Generated by {data.get('model', 'unknown')}",
            style="green"
        ))
    else:
        console.print(f"[red]Generation failed: {response.text}[/red]")

@cli.command()
def dashboard():
    """Open results dashboard in browser"""
    import webbrowser
    controller = LLMTestingCLI()
    
    results_dir = Path(controller.config.get('inference', {}).get('output_dir', 'results'))
    reports = list(results_dir.glob("report_*.html"))
    
    if reports:
        latest = sorted(reports)[-1]
        webbrowser.open(f"file://{latest.absolute()}")
        console.print(f"[green]Opened dashboard: {latest.name}[/green]")
    else:
        console.print("[yellow]No reports found. Run tests first with: llm test[/yellow]")

@cli.command()
@click.option('--output', '-o', default='config.yaml', help='Output file')
def config(output):
    """Generate default configuration file"""
    default_config = {
        "models": {
            "llama-70b": {
                "path": "/path/to/llama-3.1-70b-Instruct",
                "type": "llama",
                "tensor_parallel_size": 8,
                "max_memory_per_gpu": "80GB",
                "default_sampling": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "max_tokens": 512
                }
            }
        },
        "datasets": {
            "quick_test": [
                {"name": "sst2", "samples": 100, "metrics": ["accuracy", "valid_format_rate"]},
                {"name": "bool_logic", "samples": 100, "metrics": ["accuracy", "valid_format_rate"]}
            ]
        },
        "server": {
            "host": "0.0.0.0",
            "port": 8000,
            "timeout": 600,
            "batch_size": 32
        },
        "inference": {
            "output_dir": "results",
            "use_sampled_data": True
        }
    }
    
    with open(output, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False)
    
    console.print(f"[green]✓ Created configuration: {output}[/green]")
    console.print("[yellow]Edit the file to add your model paths[/yellow]")

if __name__ == "__main__":
    cli()