#!/usr/bin/env python3
"""
Script para preparar dados do dataset de placas.

Este script substitui o preparar_dados.py original com melhor
estrutura e tratamento de erros.
"""

import argparse
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_preparation import prepare_data_from_zips
from utils import setup_logging


def main():
    """Função principal do script."""
    parser = argparse.ArgumentParser(
        description="Preparar dados do dataset de placas",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--zips-dir",
        type=str,
        required=True,
        help="Diretório onde estão os arquivos ZIP (Treino.zip, Validacao.zip, Teste.zip)"
    )
    
    parser.add_argument(
        "--dados-dir",
        type=str,
        default="dados",
        help="Diretório de destino dos dados"
    )
    
    args = parser.parse_args()
    
    # Configurar logging
    logger = setup_logging()
    
    try:
        prepare_data_from_zips(args.zips_dir, args.dados_dir)
        logger.info("Preparação de dados concluída com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro durante preparação dos dados: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()