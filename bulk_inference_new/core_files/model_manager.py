#!/usr/bin/env python3
"""
Unified Model Manager for Multi-LLM Testing
Handles model loading, switching, and server management
"""

import yaml
import logging
import torch
import gc
import asyncio
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Model configuration dataclass"""
    name: str
    path: str
    type: str
    tensor_parallel_size: int
    max_memory_per_gpu: str
    default_sampling: Dict[str, Any]

class ModelManager:
    """Manages multiple LLM models with easy switching"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.models: Dict[str, ModelConfig] = self._parse_models()
        self.current_model: Optional[AsyncLLMEngine] = None
        self.current_model_name: Optional[str] = None
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _parse_models(self) -> Dict[str, ModelConfig]:
        """Parse model configurations"""
        models = {}
        for name, config in self.config['models'].items():
            models[name] = ModelConfig(
                name=name,
                path=config['path'],
                type=config['type'],
                tensor_parallel_size=config['tensor_parallel_size'],
                max_memory_per_gpu=config['max_memory_per_gpu'],
                default_sampling=config['default_sampling']
            )
        return models
    
    def list_models(self) -> List[str]:
        """List available models"""
        return list(self.models.keys())
    
    async def load_model(self, model_name: str) -> AsyncLLMEngine:
        """Load a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {self.list_models()}")
        
        # Unload current model if exists
        if self.current_model:
            await self.unload_current_model()
        
        model_config = self.models[model_name]
        logger.info(f"Loading model: {model_name}")
        
        # Check if model path exists
        if not Path(model_config.path).exists():
            raise FileNotFoundError(f"Model path not found: {model_config.path}")
        
        engine_args = AsyncEngineArgs(
            model=model_config.path,
            tensor_parallel_size=model_config.tensor_parallel_size,
            trust_remote_code=True,
            max_num_batched_tokens=4096,
            max_num_seqs=256,
        )
        
        self.current_model = AsyncLLMEngine.from_engine_args(engine_args)
        self.current_model_name = model_name
        logger.info(f"Model {model_name} loaded successfully")
        
        return self.current_model
    
    async def unload_current_model(self):
        """Unload the current model and free memory"""
        if self.current_model:
            logger.info(f"Unloading model: {self.current_model_name}")
            self.current_model = None
            self.current_model_name = None
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("Model unloaded and memory freed")
    
    def get_sampling_params(self, model_name: str, override: Optional[Dict] = None) -> SamplingParams:
        """Get sampling parameters for a model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        params = self.models[model_name].default_sampling.copy()
        if override:
            params.update(override)
        
        return SamplingParams(**params)

# FastAPI Application with Model Manager
class InferenceServer:
    """Enhanced inference server with model management"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.manager = ModelManager(config_path)
        self.app = FastAPI()
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.on_event("startup")
        async def startup():
            # Load default model if specified
            models = self.manager.list_models()
            if models:
                await self.manager.load_model(models[0])
        
        @self.app.on_event("shutdown")
        async def shutdown():
            await self.manager.unload_current_model()
        
        @self.app.get("/models")
        async def list_models():
            """List available models"""
            return {
                "available_models": self.manager.list_models(),
                "current_model": self.manager.current_model_name
            }
        
        @self.app.post("/load_model/{model_name}")
        async def load_model(model_name: str):
            """Load a specific model"""
            try:
                await self.manager.load_model(model_name)
                return {"status": "success", "model": model_name}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "current_model": self.manager.current_model_name,
                "model_loaded": self.manager.current_model is not None
            }
        
        @self.app.post("/generate")
        async def generate(request: GenerateRequest):
            """Generate text with current model"""
            if not self.manager.current_model:
                raise HTTPException(status_code=503, detail="No model loaded")
            
            try:
                sampling_params = self.manager.get_sampling_params(
                    self.manager.current_model_name,
                    request.sampling_params
                )
                
                req_id = str(uuid.uuid4())
                final_output = None
                
                async for output in self.manager.current_model.generate(
                    request.prompt, sampling_params, req_id
                ):
                    final_output = output
                
                if final_output and final_output.outputs:
                    return {
                        "output": final_output.outputs[0].text.strip(),
                        "model": self.manager.current_model_name
                    }
                return {"output": "", "model": self.manager.current_model_name}
                
            except Exception as e:
                logger.error(f"Generation error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/batch_generate")
        async def batch_generate(request: BatchGenerateRequest):
            """Batch generation endpoint"""
            if not self.manager.current_model:
                raise HTTPException(status_code=503, detail="No model loaded")
            
            try:
                sampling_params = self.manager.get_sampling_params(
                    self.manager.current_model_name,
                    request.sampling_params
                )
                
                tasks = []
                for prompt in request.prompts:
                    req_id = str(uuid.uuid4())
                    tasks.append(self._generate_single(prompt, sampling_params, req_id))
                
                responses = await asyncio.gather(*tasks)
                return {
                    "outputs": responses,
                    "model": self.manager.current_model_name
                }
                
            except Exception as e:
                logger.error(f"Batch generation error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def _generate_single(self, prompt: str, sampling_params: SamplingParams, req_id: str) -> str:
        """Generate single response"""
        final_output = None
        async for output in self.manager.current_model.generate(prompt, sampling_params, req_id):
            final_output = output
        
        if final_output and final_output.outputs:
            return final_output.outputs[0].text.strip()
        return ""
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the server"""
        uvicorn.run(self.app, host=host, port=port)

# Request models
class GenerateRequest(BaseModel):
    prompt: str
    sampling_params: Optional[Dict[str, Any]] = None

class BatchGenerateRequest(BaseModel):
    prompts: List[str]
    sampling_params: Optional[Dict[str, Any]] = None

# CLI for model management
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Manager CLI")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--server", action="store_true", help="Start inference server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    
    args = parser.parse_args()
    
    if args.list:
        manager = ModelManager(args.config)
        print("Available models:")
        for model in manager.list_models():
            config = manager.models[model]
            print(f"  - {model}: {config.type} at {config.path}")
    
    elif args.server:
        server = InferenceServer(args.config)
        server.run(host=args.host, port=args.port)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()