"""
Detector de Caracteres em Placas Veiculares - YOLOv9

Este pacote contém módulos para treinamento, validação e detecção
de caracteres individuais em placas veiculares brasileiras usando YOLOv9.

Módulos:
    - config: Configurações e constantes do projeto
    - data_preparation: Preparação e organização dos dados
    - training: Treinamento do modelo YOLOv9
    - validation: Validação e métricas do modelo
    - detection: Detecção e inferência em imagens
    - utils: Utilitários e funções auxiliares
"""

__version__ = "1.0.0"
__author__ = "Projeto IA Placas"
__email__ = "contato@exemplo.com"

from .config import ProjectConfig
from .utils import setup_logging, validate_paths

__all__ = [
    "ProjectConfig",
    "setup_logging", 
    "validate_paths"
]