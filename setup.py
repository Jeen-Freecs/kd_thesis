"""Setup script for Confidence-Aware Ensemble Knowledge Distillation"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Confidence-Aware Ensemble Knowledge Distillation for CIFAR-100"

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
with open(requirements_file, "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="confidence-aware-kd",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Confidence-Aware Ensemble Knowledge Distillation for Image Classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Confidence-Aware-Ensemble-Knowledge-Distillation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "kd-train=scripts.train:main",
            "kd-eval=scripts.evaluate:main",
            "kd-experiment=scripts.experiment:main",
        ],
    },
)

