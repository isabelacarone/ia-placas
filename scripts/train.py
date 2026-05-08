#!/usr/bin/env python3
"""
Script para treinamento do modelo YOLOv9.

Este script substitui o treinar.py original com melhor
estrutura e tratamento de erros.
"""

import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training import main

if __name__ == "__main__":
    main()