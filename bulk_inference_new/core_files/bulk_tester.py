#!/usr/bin/env python3
"""
Automated Bulk Testing System for Multiple LLMs
Run benchmarks across different models and datasets with one command
"""

import yaml
import json
import time
import requests
import asyncio
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from tqdm import tqdm
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import promptbench as pb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BulkTester:
    """Automated testing across multiple models and datasets"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.results_dir = Path(self.config['inference']['output_dir'])
        self.results_dir.mkdir(exist_ok=True)
        self.server_url = f"http://localhost:{self.config['server']['port']}"
        self.test_results = []
        
    def _load_config(self, path: str) -> Dict:
        """Load configuration"""
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def check_server(self) -> bool:
        """Check if inference server is running"""
        try:
            response = requests.get(f"{self.server_url}/health")
            return response.status_code == 200
        except:
            return False
    
    def load_model(self, model_name: str) -> bool:
        """Load a specific model on the server"""
        try:
            response = requests.post(f"{self.server_url}/load_model/{model_name}")
            if response.status_code == 200:
                logger.info(f"Loaded model: {model_name}")
                time.sleep(5)  # Give model time to load
                return True
            else:
                logger.error(f"Failed to load model {model_name}: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False
    
    def prepare_dataset(self, dataset_config: Dict) -> List[Dict]:
        """Prepare dataset for testing"""
        dataset_name = dataset_config['name']
        samples = dataset_config.get('samples', 100)
        subset = dataset_config.get('subset', None)
        
        try:
            # Try to load from cache first
            cache_file = self.results_dir / f"cache_{dataset_name}_{subset}_{samples}.json"
            if cache_file.exists() and self.config['inference'].get('use_sampled_data', True):
                with open(cache_file, 'r') as f:
                    logger.info(f"Loading cached dataset: {dataset_name}")
                    return json.load(f)
            
            # Load fresh dataset
            logger.info(f"Loading dataset: {dataset_name} (subset: {subset}, samples: {samples})")
            dataset = pb.DatasetLoader.load_dataset(dataset_name, subset)
            
            # Convert to list and sample
            if hasattr(dataset, '__len__'):
                dataset_list = list(dataset)[:samples]
            else:
                dataset_list = [item for i, item in enumerate(dataset) if i < samples]
            
            # Cache for future use
            with open(cache_file, 'w') as f:
                json.dump(dataset_list, f)
            
            return dataset_list
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            return []
    
    def format_prompt(self, dataset_name: str, example: Dict) -> str:
        """Simple prompt formatting"""
        # Simplified prompt formatting - you can enhance this
        if dataset_name == "sst2":
            return f"Classify the sentiment as POSITIVE or NEGATIVE: {example.get('content', '')}\nAnswer:"
        elif dataset_name == "mmlu":
            q = example.get('question', '')
            choices = example.get('choices', [])
            prompt = f"Question: {q}\n"
            for i, choice in enumerate(choices):
                prompt += f"{chr(65+i)}) {choice}\n"
            prompt += "Answer (A/B/C/D):"
            return prompt
        elif dataset_name == "bool_logic":
            return f"Evaluate as TRUE or FALSE: {example.get('content', '')}\nAnswer:"
        elif dataset_name == "valid_parentheses":
            return f"Are these parentheses VALID or INVALID: {example.get('content', '')}\nAnswer:"
        elif dataset_name.startswith("math"):
            return f"Solve: {example.get('question', '')}\nFinal Answer:"
        else:
            # Generic format
            text = example.get('content') or example.get('text') or example.get('question') or str(example)
            return f"Input: {text}\nOutput:"
    
    def evaluate_response(self, dataset_name: str, example: Dict, response: str) -> Dict[str, Any]:
        """Evaluate model response"""
        result = {
            "response": response,
            "correct": False,
            "valid_format": False
        }
        
        response = response.strip().upper()
        
        if dataset_name == "sst2":
            expected = example.get('label', -1)
            if response in ["POSITIVE", "NEGATIVE"]:
                result["valid_format"] = True
                result["correct"] = (response == "POSITIVE" and expected == 1) or \
                                  (response == "NEGATIVE" and expected == 0)
        
        elif dataset_name == "mmlu":
            expected = example.get('answer', -1)
            if response in ["A", "B", "C", "D"]:
                result["valid_format"] = True
                result["correct"] = ord(response) - 65 == expected
        
        elif dataset_name == "bool_logic":
            expected = str(example.get('label', '')).upper()
            if response in ["TRUE", "FALSE"]:
                result["valid_format"] = True
                result["correct"] = response == expected
        
        elif dataset_name == "valid_parentheses":
            expected = str(example.get('label', '')).upper()
            if response in ["VALID", "INVALID"]:
                result["valid_format"] = True
                result["correct"] = response == expected
        
        return result
    
    def test_model_on_dataset(
        self, 
        model_name: str, 
        dataset_config: Dict,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """Test a model on a specific dataset"""
        dataset_name = dataset_config['name']
        logger.info(f"Testing {model_name} on {dataset_name}")
        
        # Prepare dataset
        dataset = self.prepare_dataset(dataset_config)
        if not dataset:
            return {"error": "Failed to load dataset"}
        
        results = []
        total_time = 0
        
        # Process in batches
        for i in tqdm(range(0, len(dataset), batch_size), desc=f"{model_name}/{dataset_name}"):
            batch = dataset[i:i+batch_size]
            prompts = [self.format_prompt(dataset_name, ex) for ex in batch]
            
            # Send batch request
            start_time = time.time()
            try:
                response = requests.post(
                    f"{self.server_url}/batch_generate",
                    json={"prompts": prompts},
                    timeout=300
                )
                
                if response.status_code == 200:
                    outputs = response.json()['outputs']
                    batch_time = time.time() - start_time
                    total_time += batch_time
                    
                    # Evaluate responses
                    for example, output in zip(batch, outputs):
                        eval_result = self.evaluate_response(dataset_name, example, output)
                        eval_result['latency'] = batch_time / len(batch)
                        results.append(eval_result)
                else:
                    logger.error(f"Batch request failed: {response.status_code}")
                    for example in batch:
                        results.append({"error": "request_failed"})
                        
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                for example in batch:
                    results.append({"error": str(e)})
        
        # Calculate metrics
        valid_results = [r for r in results if 'error' not in r]
        metrics = {
            "model": model_name,
            "dataset": dataset_name,
            "total_examples": len(dataset),
            "successful_examples": len(valid_results),
            "accuracy": sum(r['correct'] for r in valid_results) / len(valid_results) * 100 if valid_results else 0,
            "valid_format_rate": sum(r['valid_format'] for r in valid_results) / len(valid_results) * 100 if valid_results else 0,
            "avg_latency": sum(r.get('latency', 0) for r in valid_results) / len(valid_results) if valid_results else 0,
            "total_time": total_time,
            "errors": len(results) - len(valid_results)
        }
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = self.results_dir / f"{model_name}_{dataset_name}_{timestamp}.json"
        with open(result_file, 'w') as f:
            json.dump({
                "metrics": metrics,
                "detailed_results": results,
                "config": dataset_config
            }, f, indent=2)
        
        return metrics
    
    def run_benchmark(
        self,
        models: Optional[List[str]] = None,
        dataset_suite: str = "quick_test"
    ) -> pd.DataFrame:
        """Run complete benchmark across models and datasets"""
        
        # Check server
        if not self.check_server():
            logger.error("Inference server not running. Start it with: python model_manager.py --server")
            return pd.DataFrame()
        
        # Get models to test
        if models is None:
            models = list(self.config['models'].keys())
        
        # Get datasets to test
        datasets = self.config['datasets'].get(dataset_suite, [])
        if not datasets:
            logger.error(f"Dataset suite '{dataset_suite}' not found")
            return pd.DataFrame()
        
        logger.info(f"Starting benchmark: {len(models)} models × {len(datasets)} datasets")
        
        all_results = []
        
        # Test each model
        for model_name in models:
            logger.info(f"\n{'='*50}")
            logger.info(f"Testing model: {model_name}")
            logger.info(f"{'='*50}")
            
            # Load model
            if not self.load_model(model_name):
                logger.error(f"Failed to load {model_name}, skipping...")
                continue
            
            # Test on each dataset
            for dataset_config in datasets:
                metrics = self.test_model_on_dataset(
                    model_name,
                    dataset_config,
                    batch_size=self.config['server']['batch_size']
                )
                all_results.append(metrics)
                
                # Log intermediate results
                logger.info(f"  {dataset_config['name']}: "
                          f"Acc={metrics.get('accuracy', 0):.1f}%, "
                          f"Valid={metrics.get('valid_format_rate', 0):.1f}%, "
                          f"Latency={metrics.get('avg_latency', 0):.3f}s")
        
        # Create results dataframe
        df = pd.DataFrame(all_results)
        
        # Save summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.results_dir / f"benchmark_summary_{timestamp}.csv"
        df.to_csv(summary_file, index=False)
        
        # Print summary
        print("\n" + "="*70)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*70)
        print(df.to_string())
        
        # Create comparison report
        self.create_comparison_report(df, timestamp)
        
        return df
    
    def create_comparison_report(self, df: pd.DataFrame, timestamp: str):
        """Create HTML comparison report"""
        html_file = self.results_dir / f"report_{timestamp}.html"
        
        # Pivot table for accuracy
        if not df.empty and 'accuracy' in df.columns:
            pivot_acc = df.pivot_table(
                values='accuracy',
                index='dataset',
                columns='model',
                aggfunc='mean'
            )
            
            pivot_format = df.pivot_table(
                values='valid_format_rate',
                index='dataset',
                columns='model',
                aggfunc='mean'
            )
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Benchmark Report - {timestamp}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                    th {{ background-color: #4CAF50; color: white; }}
                    tr:nth-child(even) {{ background-color: #f2f2f2; }}
                    .best {{ background-color: #90EE90; font-weight: bold; }}
                </style>
            </head>
            <body>
                <h1>LLM Benchmark Report</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>Accuracy Comparison (%)</h2>
                {pivot_acc.to_html(float_format=lambda x: f'{x:.1f}')}
                
                <h2>Valid Format Rate (%)</h2>
                {pivot_format.to_html(float_format=lambda x: f'{x:.1f}')}
                
                <h2>Detailed Results</h2>
                {df.to_html(index=False, float_format=lambda x: f'{x:.2f}' if isinstance(x, float) else x)}
            </body>
            </html>
            """
            
            with open(html_file, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Report saved to: {html_file}")

def main():
    parser = argparse.ArgumentParser(description="Automated Multi-Model Testing")
    parser.add_argument("--config", default="config.yaml", help="Configuration file")
    parser.add_argument("--models", nargs="+", help="Models to test (default: all)")
    parser.add_argument("--suite", default="quick_test", 
                       choices=["quick_test", "full_benchmark", "translation_test"],
                       help="Dataset suite to run")
    parser.add_argument("--compare", action="store_true", help="Compare results from previous runs")
    
    args = parser.parse_args()
    
    tester = BulkTester(args.config)
    
    if args.compare:
        # Load and compare previous results
        result_files = list(Path(tester.results_dir).glob("benchmark_summary_*.csv"))
        if len(result_files) >= 2:
            latest_files = sorted(result_files)[-2:]
            df1 = pd.read_csv(latest_files[0])
            df2 = pd.read_csv(latest_files[1])
            
            print("\nComparing last two runs:")
            print(f"Run 1: {latest_files[0].name}")
            print(f"Run 2: {latest_files[1].name}")
            
            # Compare accuracy
            for model in df2['model'].unique():
                print(f"\n{model}:")
                for dataset in df2['dataset'].unique():
                    acc1 = df1[(df1['model'] == model) & (df1['dataset'] == dataset)]['accuracy'].values
                    acc2 = df2[(df2['model'] == model) & (df2['dataset'] == dataset)]['accuracy'].values
                    if acc1.size > 0 and acc2.size > 0:
                        diff = acc2[0] - acc1[0]
                        symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
                        print(f"  {dataset}: {acc1[0]:.1f}% → {acc2[0]:.1f}% ({symbol}{abs(diff):.1f}%)")
        else:
            print("Not enough previous runs to compare")
    else:
        # Run new benchmark
        results = tester.run_benchmark(
            models=args.models,
            dataset_suite=args.suite
        )

if __name__ == "__main__":
    main()