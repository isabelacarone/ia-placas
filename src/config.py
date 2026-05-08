"""
Configurações centralizadas do projeto.

Este módulo contém todas as configurações, constantes e caminhos
utilizados no projeto de detecção de placas.
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class ProjectConfig:
    """Configurações centralizadas do projeto."""
    
    # Diretórios principais
    ROOT_DIR: Path = Path(__file__).parent.parent
    SRC_DIR: Path = ROOT_DIR / "src"
    CONFIGS_DIR: Path = ROOT_DIR / "configs"
    DATA_DIR: Path = ROOT_DIR / "dados"
    WEIGHTS_DIR: Path = ROOT_DIR / "pesos"
    YOLOV9_DIR: Path = ROOT_DIR / "uvv" / "yolov9-main"
    
    # Arquivos de configuração
    DATA_YAML: Path = CONFIGS_DIR / "dados-placas.yaml"
    HYPERPARAMS_YAML: Path = CONFIGS_DIR / "hyp.scratch-high.yaml"
    MODEL_YAML: Path = YOLOV9_DIR / "models" / "detect" / "yolov9-c.yaml"
    
    # Diretórios de dados
    TRAIN_DIR: Path = DATA_DIR / "treino"
    VAL_DIR: Path = DATA_DIR / "validacao"
    TEST_DIR: Path = DATA_DIR / "teste"
    
    # Scripts do YOLOv9
    TRAIN_SCRIPT: Path = YOLOV9_DIR / "train_dual.py"
    VAL_SCRIPT: Path = YOLOV9_DIR / "val.py"
    DETECT_SCRIPT: Path = YOLOV9_DIR / "detect.py"
    
    # Classes detectadas
    NUM_CLASSES: int = 35
    CLASS_NAMES: List[str] = None
    
    # Configurações de treinamento padrão
    DEFAULT_EPOCHS: int = 300
    DEFAULT_BATCH_SIZE: int = 8
    DEFAULT_IMG_SIZE: int = 640
    DEFAULT_WORKERS: int = 0
    
    # Configurações de detecção padrão
    DEFAULT_CONF_THRESHOLD: float = 0.25
    DEFAULT_IOU_THRESHOLD: float = 0.45
    
    # Configurações de validação padrão
    DEFAULT_VAL_CONF: float = 0.001
    DEFAULT_VAL_IOU: float = 0.7
    DEFAULT_VAL_BATCH: int = 32
    
    def __post_init__(self):
        """Inicializa configurações após criação da instância."""
        if self.CLASS_NAMES is None:
            self.CLASS_NAMES = self._get_class_names()
        
        # Configurar variáveis de ambiente
        self._setup_environment()
    
    def _get_class_names(self) -> List[str]:
        """Retorna lista com nomes das classes."""
        digits = [str(i) for i in range(10)]
        letters = [chr(i) for i in range(ord('A'), ord('Z') + 1) if chr(i) != 'O']
        return digits + letters
    
    def _setup_environment(self) -> None:
        """Configura variáveis de ambiente necessárias."""
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["WANDB_DISABLED"] = "true"
    
    def create_directories(self) -> None:
        """Cria diretórios necessários se não existirem."""
        directories = [
            self.DATA_DIR,
            self.TRAIN_DIR,
            self.VAL_DIR, 
            self.TEST_DIR,
            self.WEIGHTS_DIR,
            self.CONFIGS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def validate_paths(self) -> Dict[str, bool]:
        """Valida se os caminhos essenciais existem."""
        paths_to_check = {
            "YOLOv9 Directory": self.YOLOV9_DIR,
            "Train Script": self.TRAIN_SCRIPT,
            "Validation Script": self.VAL_SCRIPT,
            "Detection Script": self.DETECT_SCRIPT,
            "Data YAML": self.DATA_YAML,
            "Hyperparams YAML": self.HYPERPARAMS_YAML,
            "Model YAML": self.MODEL_YAML
        }
        
        return {name: path.exists() for name, path in paths_to_check.items()}


# Instância global de configuração
config = ProjectConfig()