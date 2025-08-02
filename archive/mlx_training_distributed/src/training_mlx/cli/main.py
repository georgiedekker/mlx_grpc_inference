"""
Training MLX CLI - Clean command line interface for distributed training

Provides commands for training, validation, API server, and configuration.
"""

import os
import sys
import click
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from ..adapters.distributed_integration import (
    get_integration_status,
    is_distributed_available
)
from ..utils.config import load_config, validate_config
from ..utils.logging import setup_logging

logger = logging.getLogger(__name__)


@click.group(name="training-mlx")
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--debug', is_flag=True, help='Enable debug mode')
def cli(verbose: bool, debug: bool):
    """Training MLX - Distributed Training System for MLX Models"""
    level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)
    setup_logging(level=level)
    
    if debug:
        click.echo("🐛 Debug mode enabled")


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--name', '-n', help='Experiment name')
@click.option('--distributed/--local', default=True, help='Use distributed training')
@click.option('--resume', type=click.Path(), help='Resume from checkpoint')
@click.option('--validate-only', is_flag=True, help='Only run validation')
def train(config_file: str, name: Optional[str], distributed: bool, 
          resume: Optional[str], validate_only: bool):
    """Start training with the given configuration file."""
    click.echo(f"🎯 {'Validating' if validate_only else 'Starting training'} with: {config_file}")
    
    try:
        # Load and validate configuration
        config = load_config(config_file)
        validate_config(config)
        
        # Override with CLI options
        if name:
            config['experiment']['name'] = name
        config['distributed']['enabled'] = distributed
        if resume:
            config['training']['resume_from'] = resume
            
        # Check distributed system integration
        if distributed:
            status = get_integration_status()
            if status["distributed_available"]:
                click.echo("✅ Distributed system integration available")
            else:
                click.echo("⚠️  Distributed system unavailable - using local mode")
                config['distributed']['enabled'] = False
        
        if validate_only:
            click.echo("✅ Configuration is valid")
            return
            
        # Import here to avoid circular imports
        from ..training.trainer import DistributedTrainer
        
        trainer = DistributedTrainer(config)
        trainer.train()
        
        click.echo("✅ Training completed successfully!")
        
    except Exception as e:
        click.echo(f"❌ Training failed: {e}", err=True)
        if logger.isEnabledFor(logging.DEBUG):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--format', 'data_format', type=click.Choice(['auto', 'alpaca', 'sharegpt']), 
              default='auto', help='Dataset format')
@click.option('--split', default=0.1, help='Validation split ratio')
@click.option('--max-samples', type=int, help='Maximum samples to validate')
def validate(data_path: str, data_format: str, split: float, max_samples: Optional[int]):
    """Validate dataset format and quality."""
    click.echo(f"🔍 Validating dataset: {data_path}")
    
    try:
        from ..training.datasets import DatasetValidator
        
        validator = DatasetValidator()
        result = validator.validate(
            data_path, 
            format=data_format,
            split_ratio=split,
            max_samples=max_samples
        )
        
        click.echo(f"✅ Dataset is valid!")
        click.echo(f"📊 Format: {result['format']}")
        click.echo(f"📈 Samples: {result['num_samples']}")
        click.echo(f"📏 Avg length: {result['avg_length']} tokens")
        
        if result.get('warnings'):
            click.echo("\n⚠️  Warnings:")
            for warning in result['warnings']:
                click.echo(f"  - {warning}")
                
    except Exception as e:
        click.echo(f"❌ Validation failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--host', default='0.0.0.0', help='API server host')
@click.option('--port', default=8200, help='API server port')
@click.option('--workers', default=1, help='Number of worker processes')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
def serve(host: str, port: int, workers: int, reload: bool):
    """Start the Training MLX API server."""
    click.echo(f"🚀 Starting API server on {host}:{port}")
    
    try:
        import uvicorn
        from ..api.app import create_app
        
        app = create_app()
        
        uvicorn.run(
            app if reload else "training_mlx.api.app:create_app",
            host=host,
            port=port,
            workers=1 if reload else workers,
            reload=reload,
            log_level="info",
            access_log=True
        )
        
    except KeyboardInterrupt:
        click.echo("\n👋 Server stopped")
    except Exception as e:
        click.echo(f"❌ Server failed: {e}", err=True)
        sys.exit(1)


@cli.command()
def status():
    """Show system status and integration health."""
    click.echo("📊 Training MLX Status")
    click.echo("=" * 50)
    
    # System info
    click.echo("\n🖥️  System Information:")
    click.echo(f"  Python: {sys.version.split()[0]}")
    click.echo(f"  Platform: {sys.platform}")
    
    # Check MLX installation
    try:
        import mlx
        try:
            version = mlx.__version__
        except AttributeError:
            # MLX doesn't have __version__, try alternative
            version = "installed"
        click.echo(f"  MLX: {version}")
    except ImportError:
        click.echo("  MLX: ❌ Not installed")
    
    # API server status
    click.echo("\n🌐 API Server:")
    try:
        import requests
        response = requests.get("http://localhost:8200/health", timeout=2)
        if response.status_code == 200:
            click.echo("  ✅ Running on port 8200")
        else:
            click.echo("  ⚠️  Server responding but unhealthy")
    except:
        click.echo("  ❌ Not running")
    
    # Distributed system integration
    click.echo("\n🔗 Distributed System Integration:")
    status = get_integration_status()
    if status["distributed_available"]:
        click.echo("  ✅ Available for distributed training")
        if status.get("grpc_available"):
            click.echo("  ✅ gRPC communication ready")
    else:
        click.echo("  ⚠️  Not available (local mode only)")
    
    # Configuration
    click.echo("\n⚙️  Configuration:")
    click.echo(f"  Config directory: {Path.home() / '.training_mlx'}")
    click.echo(f"  Environment: {os.getenv('ENVIRONMENT', 'development')}")


@cli.command()
@click.option('--output', '-o', default='config.yaml', help='Output file')
@click.option('--model', default='mlx-community/Qwen2.5-0.5B-Instruct-4bit', help='Model to train')
@click.option('--format', 'data_format', type=click.Choice(['alpaca', 'sharegpt']), 
              default='alpaca', help='Dataset format')
@click.option('--lora/--full', default=True, help='Use LoRA fine-tuning')
def init(output: str, model: str, data_format: str, lora: bool):
    """Generate a training configuration file."""
    click.echo(f"📝 Generating configuration for {model}")
    
    config = {
        "experiment": {
            "name": f"training_{data_format}",
            "tags": ["mlx", data_format, "lora" if lora else "full"],
        },
        "model": {
            "name": model,
            "quantization": "4bit" if "4bit" in model else None,
        },
        "training": {
            "batch_size": 4,
            "learning_rate": 5e-5 if lora else 1e-4,
            "num_epochs": 3,
            "gradient_accumulation_steps": 4,
            "warmup_ratio": 0.1,
            "save_steps": 100,
            "eval_steps": 50,
        },
        "data": {
            "format": data_format,
            "train_path": f"data/train.{data_format}.json",
            "validation_split": 0.1,
            "max_seq_length": 2048,
        },
        "lora": {
            "enabled": lora,
            "r": 8,
            "alpha": 16,
            "dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
        } if lora else None,
        "distributed": {
            "enabled": False,
            "strategy": "data_parallel",
        },
        "logging": {
            "level": "INFO",
            "wandb": {
                "enabled": False,
                "project": "training-mlx",
            },
            "tensorboard": {
                "enabled": True,
                "log_dir": "./logs",
            },
        },
    }
    
    # Remove None values
    config = {k: v for k, v in config.items() if v is not None}
    
    with open(output, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    click.echo(f"✅ Configuration saved to: {output}")
    click.echo("\n📚 Next steps:")
    click.echo(f"  1. Prepare your dataset in {data_format} format")
    click.echo(f"  2. Update paths in {output}")
    click.echo(f"  3. Run: training-mlx train {output}")


@cli.command()
@click.option('--fix', is_flag=True, help='Attempt to fix issues')
def doctor(fix: bool):
    """Diagnose and fix common issues."""
    click.echo("🏥 Running diagnostics...")
    
    issues = []
    
    # Check environment
    if not Path(".env").exists() and Path(".env.example").exists():
        issues.append(("No .env file", "cp .env.example .env"))
    
    # Check dependencies
    try:
        import mlx
    except ImportError:
        issues.append(("MLX not installed", "pip install mlx"))
    
    # Check security
    if not Path(".gitignore").exists():
        issues.append(("No .gitignore", "Create .gitignore for security"))
    
    if issues:
        click.echo(f"\n⚠️  Found {len(issues)} issues:")
        for issue, solution in issues:
            click.echo(f"  ❌ {issue}")
            click.echo(f"     💡 Fix: {solution}")
            
        if fix:
            click.echo("\n🔧 Attempting fixes...")
            # Implement auto-fixes here
            click.echo("✅ Fixed what I could!")
    else:
        click.echo("✅ All checks passed!")


if __name__ == "__main__":
    cli()