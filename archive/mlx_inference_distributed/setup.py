"""
Setup configuration for mlx-distributed package.

This package provides distributed inference capabilities for MLX models
across multiple Apple Silicon devices.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mlx-distributed",
    version="1.0.0",
    author="MLX Distributed Team",
    author_email="team@example.com",
    description="Distributed inference for MLX models across Apple Silicon devices",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/mlx-distributed",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: MacOS :: MacOS X",
    ],
    python_requires=">=3.9",
    install_requires=[
        "mlx>=0.27.0",
        "mlx-lm>=0.26.0",
        "grpcio>=1.60.0",
        "grpcio-tools>=1.60.0",
        "protobuf>=4.0.0",
        "numpy>=1.24.0",
        "psutil>=5.9.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "httpx>=0.24.0",
        "aiohttp>=3.8.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "isort>=5.12.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "sphinx-autodoc-typehints>=1.22.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mlx-distributed=mlx_distributed.cli:main",
            "mlx-dist-server=mlx_distributed.distributed_api:main",
            "mlx-dist-worker=mlx_distributed.worker:main",
            "mlx-dist-detect=mlx_distributed.hardware_detector:main",
            "mlx-dist-config=mlx_distributed.auto_configure_cluster:main",
        ],
    },
    package_data={
        "mlx_distributed": [
            "protos/*.proto",
            "configs/*.json",
            "scripts/*.sh",
        ],
    },
    include_package_data=True,
)