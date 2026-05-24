"""
Fixtures compartilhadas para os testes do projeto detector-placas-yolov9.
"""

import pytest
from pathlib import Path
from src.detection import PlateCharacterProcessor


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Fornece um diretório temporário para os testes."""
    return tmp_path


@pytest.fixture
def sample_yolo_label_file(tmp_path: Path) -> Path:
    """
    Cria um arquivo de label YOLO de exemplo com linhas válidas de 5 e 6 campos.

    Formato YOLO:
        class_id  x_center  y_center  width  height  [confidence]

    Retorna o caminho para o arquivo criado.
    """
    label_file = tmp_path / "sample_label.txt"
    lines = [
        # 5 campos (sem confidence)
        "0 0.5 0.5 0.2 0.3\n",
        "1 0.3 0.4 0.15 0.25\n",
        # 6 campos (com confidence)
        "10 0.7 0.6 0.18 0.28 0.92\n",
        "25 0.2 0.5 0.12 0.22 0.85\n",
        # Linha inválida (deve ser ignorada pelo parser)
        "invalid line here\n",
    ]
    label_file.write_text("".join(lines), encoding="utf-8")
    return label_file


@pytest.fixture
def sample_detections() -> list:
    """
    Fornece uma lista de dicionários de detecção de exemplo.

    Cada dicionário segue o formato Detection Dict definido no design:
        class_id, class_name, x_center, y_center, width, height, confidence
    """
    return [
        {
            "class_id": 0,
            "class_name": "0",
            "x_center": 0.1,
            "y_center": 0.5,
            "width": 0.05,
            "height": 0.2,
            "confidence": 0.95,
        },
        {
            "class_id": 1,
            "class_name": "1",
            "x_center": 0.2,
            "y_center": 0.5,
            "width": 0.05,
            "height": 0.2,
            "confidence": 0.88,
        },
        {
            "class_id": 10,
            "class_name": "A",
            "x_center": 0.35,
            "y_center": 0.5,
            "width": 0.06,
            "height": 0.22,
            "confidence": 0.76,
        },
        {
            "class_id": 11,
            "class_name": "B",
            "x_center": 0.5,
            "y_center": 0.5,
            "width": 0.06,
            "height": 0.22,
            "confidence": 0.91,
        },
        {
            "class_id": 12,
            "class_name": "C",
            "x_center": 0.65,
            "y_center": 0.5,
            "width": 0.06,
            "height": 0.22,
            "confidence": 0.83,
        },
        {
            "class_id": 2,
            "class_name": "2",
            "x_center": 0.78,
            "y_center": 0.5,
            "width": 0.05,
            "height": 0.2,
            "confidence": 0.79,
        },
        {
            "class_id": 3,
            "class_name": "3",
            "x_center": 0.9,
            "y_center": 0.5,
            "width": 0.05,
            "height": 0.2,
            "confidence": 0.87,
        },
    ]


@pytest.fixture
def plate_processor() -> PlateCharacterProcessor:
    """Fornece uma instância de PlateCharacterProcessor para os testes."""
    return PlateCharacterProcessor()
