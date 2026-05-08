"""
Setup script para o Detector de Placas YOLOv9.

Este script permite instalar o projeto como um pacote Python,
facilitando imports e distribuição.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Ler README
readme_path = Path(__file__).parent / "README_REFATORADO.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Ler requirements
requirements_path = Path(__file__).parent / "requirements_refatorado.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, 'r', encoding='utf-8') as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith('#') and not line.startswith('-')
        ]

setup(
    name="detector-placas-yolov9",
    version="2.0.0",
    author="Projeto IA Placas",
    author_email="contato@exemplo.com",
    description="Detector de caracteres em placas veiculares usando YOLOv9",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/usuario/detector-placas-yolov9",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "isort>=5.10.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
        "web": [
            "flask>=2.2.0",
            "flask-cors>=3.0.0",
        ],
        "api": [
            "fastapi>=0.85.0",
            "uvicorn>=0.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "placas-prepare=scripts.prepare_data:main",
            "placas-train=scripts.train:main",
            "placas-validate=scripts.validate:main",
            "placas-detect=scripts.detect:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.md", "*.txt"],
    },
    keywords=[
        "yolo",
        "yolov9", 
        "object-detection",
        "license-plate",
        "character-recognition",
        "computer-vision",
        "deep-learning",
        "pytorch",
    ],
    project_urls={
        "Bug Reports": "https://github.com/usuario/detector-placas-yolov9/issues",
        "Source": "https://github.com/usuario/detector-placas-yolov9",
        "Documentation": "https://github.com/usuario/detector-placas-yolov9/docs",
    },
)