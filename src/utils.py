"""
Utilitários e funções auxiliares do projeto.

Este módulo contém funções utilitárias para logging, validação,
formatação e outras operações comuns.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

try:
    from .config import config
except ImportError:
    # Para execução standalone
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from config import config


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Configura o sistema de logging do projeto.
    
    Args:
        level: Nível de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Arquivo para salvar logs (opcional)
        format_string: Formato personalizado das mensagens
    
    Returns:
        Logger configurado
    """
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    # Configurar formatador
    formatter = logging.Formatter(format_string)
    
    # Configurar logger principal
    logger = logging.getLogger("placas_detector")
    logger.setLevel(level)
    
    # Remover handlers existentes
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler para arquivo (se especificado)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def validate_paths(paths: Dict[str, Path]) -> Dict[str, bool]:
    """
    Valida se os caminhos especificados existem.
    
    Args:
        paths: Dicionário com nome e caminho para validar
    
    Returns:
        Dicionário com resultado da validação
    """
    return {name: path.exists() for name, path in paths.items()}


def print_header(title: str, width: int = 60) -> None:
    """
    Imprime cabeçalho formatado.
    
    Args:
        title: Título do cabeçalho
        width: Largura do cabeçalho
    """
    print("=" * width)
    print(f"{title:^{width}}")
    print("=" * width)


def print_config_summary(
    epochs: int,
    batch_size: int,
    img_size: int,
    device: str,
    name: str,
    **kwargs
) -> None:
    """
    Imprime resumo das configurações.
    
    Args:
        epochs: Número de épocas
        batch_size: Tamanho do batch
        img_size: Tamanho da imagem
        device: Dispositivo (GPU/CPU)
        name: Nome do experimento
        **kwargs: Configurações adicionais
    """
    print(f"  Épocas:     {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Img size:   {img_size}")
    print(f"  Device:     {device}")
    print(f"  Nome:       {name}")
    
    for key, value in kwargs.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")


def format_file_count(directory: Path, extensions: List[str]) -> str:
    """
    Conta arquivos com extensões específicas em um diretório.
    
    Args:
        directory: Diretório para buscar
        extensions: Lista de extensões (ex: ['.jpg', '.png'])
    
    Returns:
        String formatada com contagem
    """
    if not directory.exists():
        return "0 arquivos (diretório não existe)"
    
    count = 0
    for ext in extensions:
        count += len(list(directory.rglob(f"*{ext}")))
    
    return f"{count} arquivos"


def get_timestamp() -> str:
    """
    Retorna timestamp formatado para nomes de arquivos.
    
    Returns:
        String com timestamp (YYYYMMDD_HHMMSS)
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def validate_device(device: str) -> str:
    """
    Valida e normaliza especificação de dispositivo.
    
    Args:
        device: Dispositivo especificado ('0', 'cpu', 'cuda:0', etc.)
    
    Returns:
        Dispositivo normalizado
    
    Raises:
        ValueError: Se dispositivo inválido
    """
    device = device.lower().strip()
    
    if device == "cpu":
        return "cpu"
    
    if device.isdigit():
        return device
    
    if device.startswith("cuda:") and device[5:].isdigit():
        return device[5:]  # Retorna apenas o número
    
    raise ValueError(f"Dispositivo inválido: '{device}'")


def check_yolov9_installation() -> bool:
    """
    Verifica se o YOLOv9 está instalado corretamente.
    
    Returns:
        True se instalação válida, False caso contrário
    """
    required_paths = [
        config.YOLOV9_DIR,
        config.TRAIN_SCRIPT,
        config.VAL_SCRIPT,
        config.DETECT_SCRIPT
    ]
    
    return all(path.exists() for path in required_paths)


def get_class_mapping() -> Dict[int, str]:
    """
    Retorna mapeamento de índices para nomes de classes.
    
    Returns:
        Dicionário {índice: nome_classe}
    """
    return {i: name for i, name in enumerate(config.CLASS_NAMES)}


def parse_yolo_label(label_path: Path) -> List[Dict[str, Union[int, float]]]:
    """
    Parseia arquivo de label do YOLO.
    
    Args:
        label_path: Caminho para arquivo .txt
    
    Returns:
        Lista de dicionários com informações das detecções
    """
    if not label_path.exists():
        return []
    
    detections = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                detection = {
                    'class_id': int(parts[0]),
                    'x_center': float(parts[1]),
                    'y_center': float(parts[2]),
                    'width': float(parts[3]),
                    'height': float(parts[4])
                }
                detections.append(detection)
    
    return detections