"""
Elastic-Guardian X: Production-Grade AI Training Operating System
Setup configuration
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read version from package
version_file = Path(__file__).parent / "elastic_guardian" / "version.py"
version = {}
if version_file.exists():
    exec(version_file.read_text(), version)
else:
    version["__version__"] = "0.1.0"

# Core dependencies
install_requires = [
    # Deep Learning Frameworks
    "torch>=2.1.0",
    "transformers>=4.35.0",
    "accelerate>=0.25.0",
    
    # Configuration & Validation
    "pydantic>=2.0.0",
    "pyyaml>=6.0",
    "jsonschema>=4.0.0",
    
    # Hardware Monitoring
    "psutil>=5.9.0",
    "pynvml>=11.5.0",
    
    # Data Processing
    "numpy>=1.24.0",
    "datasets>=2.14.0",
    "tokenizers>=0.15.0",
    
    # Utilities
    "tqdm>=4.65.0",
    "rich>=13.0.0",  # Beautiful terminal output
    "click>=8.1.0",  # CLI framework (for future)
    
    # Serialization
    "safetensors>=0.4.0",
    
    # Logging & Monitoring
    "tensorboard>=2.15.0",
    "python-json-logger>=2.0.0",
]

# Optional dependencies for advanced features
extras_require = {
    # Development dependencies
    "dev": [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "pytest-xdist>=3.3.0",  # Parallel testing
        "pytest-timeout>=2.1.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "mypy>=1.5.0",
        "isort>=5.12.0",
        "pre-commit>=3.3.0",
    ],
    
    # Documentation
    "docs": [
        "sphinx>=7.0.0",
        "sphinx-rtd-theme>=1.3.0",
        "sphinx-autodoc-typehints>=1.24.0",
        "myst-parser>=2.0.0",
    ],
    
    # Distributed Training
    "distributed": [
        "deepspeed>=0.12.0",
        "fairscale>=0.4.13",
    ],
    
    # Advanced Optimization
    "optimization": [
        "bitsandbytes>=0.41.0",  # Quantization
        "peft>=0.7.0",  # Parameter-efficient fine-tuning
        "flash-attn>=2.3.0",  # Flash Attention (if available)
    ],
    
    # Monitoring & Tracking
    "monitoring": [
        "wandb>=0.16.0",
        "mlflow>=2.9.0",
        "prometheus-client>=0.19.0",
    ],
    
    # Cloud Storage
    "cloud": [
        "boto3>=1.28.0",  # AWS S3
        "google-cloud-storage>=2.10.0",  # GCS
        "azure-storage-blob>=12.19.0",  # Azure Blob
    ],
}

# All extra dependencies combined
extras_require["all"] = [
    dep for deps in extras_require.values() for dep in deps
]

setup(
    name="elastic-guardian",
    version=version.get("__version__", "0.1.0"),
    
    # Package information
    author="Elastic-Guardian X Team",
    author_email="elastic-guardian@example.com",
    description="Production-grade AI training operating system with zero-failure guarantees",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/elastic-guardian-x",
    
    # Package discovery
    packages=find_packages(exclude=["tests", "tests.*", "docs", "examples"]),
    
    # Python version requirement
    python_requires=">=3.10",
    
    # Dependencies
    install_requires=install_requires,
    extras_require=extras_require,
    
    # Package data
    include_package_data=True,
    package_data={
        "elastic_guardian": [
            "configs/*.yaml",
            "configs/schemas/*.json",
        ],
    },
    
    # Entry points for CLI (future)
    entry_points={
        "console_scripts": [
            # Will be added when CLI is implemented
            # "elastic-train=elastic_guardian.cli.train:main",
        ],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
    ],
    
    # Keywords
    keywords=[
        "ai", "machine-learning", "deep-learning", "training",
        "pytorch", "transformers", "distributed-training",
        "memory-management", "oom-prevention", "production-ml"
    ],
    
    # License
    license="Apache 2.0",
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/your-org/elastic-guardian-x/issues",
        "Source": "https://github.com/your-org/elastic-guardian-x",
        "Documentation": "https://elastic-guardian-x.readthedocs.io",
    },
)
