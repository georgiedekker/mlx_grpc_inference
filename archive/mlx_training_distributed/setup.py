from setuptools import setup, find_packages

setup(
    name="mlx-training-distributed",
    version="2.0.0",
    description="Production-ready MLX training framework with distributed support",
    author="MLX Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "mlx>=0.5.0",
        "mlx-lm>=0.5.0",
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.23.0",
        "pydantic>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyarrow>=12.0.0",  # For parquet support
        "aiofiles>=23.0.0",
        "python-multipart>=0.0.6",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.4.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "mlx-train=training.cli:main",
            "mlx-server=training.server:main",
        ]
    }
)