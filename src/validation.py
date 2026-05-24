"""
Módulo de validação do modelo YOLOv9.

Este módulo contém classes e funções para validar o modelo treinado
e gerar métricas de performance.
"""

import sys
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
import argparse

try:
    from .config import config
    from .utils import (
        setup_logging,
        validate_device,
        print_header,
        print_config_summary,
        check_yolov9_installation
    )
except ImportError:
    # Para execução standalone
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from config import config
    from utils import (
        setup_logging,
        validate_device,
        print_header,
        print_config_summary,
        check_yolov9_installation
    )


logger = setup_logging()


class YOLOv9Validator:
    """Classe para validação do modelo YOLOv9."""
    
    def __init__(self):
        """Inicializa o validador."""
        self.config = config
        self._validate_setup()
    
    def _validate_setup(self) -> None:
        """Valida se o ambiente está configurado corretamente."""
        if not check_yolov9_installation():
            raise RuntimeError(
                "YOLOv9 não está instalado corretamente. "
                "Verifique se a pasta uvv/yolov9-main/ existe."
            )
        
        if not self.config.DATA_YAML.exists():
            raise FileNotFoundError(
                f"Arquivo de configuração não encontrado: {self.config.DATA_YAML}"
            )
    
    def validate(
        self,
        weights: str,
        img_size: int = None,
        device: str = "0",
        conf_threshold: float = None,
        iou_threshold: float = None,
        name: str = "validacao_placas",
        batch_size: int = None,
        verbose: bool = False,
        **kwargs
    ) -> int:
        """
        Executa validação do modelo.
        
        Args:
            weights: Caminho para os pesos do modelo (.pt)
            img_size: Tamanho da imagem (padrão: config.DEFAULT_IMG_SIZE)
            device: Dispositivo ('0' para GPU, 'cpu' para CPU)
            conf_threshold: Threshold de confiança (padrão: config.DEFAULT_VAL_CONF)
            iou_threshold: Threshold de IoU (padrão: config.DEFAULT_VAL_IOU)
            name: Nome do experimento
            batch_size: Tamanho do batch (padrão: config.DEFAULT_VAL_BATCH)
            verbose: Mostrar métricas por classe
            **kwargs: Argumentos adicionais
        
        Returns:
            Código de retorno do processo de validação
        
        Raises:
            FileNotFoundError: Se pesos não encontrados
            ValueError: Se parâmetros inválidos
        """
        # Validar arquivo de pesos
        weights_path = Path(weights)
        if not weights_path.exists():
            raise FileNotFoundError(f"Pesos não encontrados: {weights_path}")
        
        # Usar valores padrão se não especificados
        img_size = img_size or self.config.DEFAULT_IMG_SIZE
        conf_threshold = conf_threshold or self.config.DEFAULT_VAL_CONF
        iou_threshold = iou_threshold or self.config.DEFAULT_VAL_IOU
        batch_size = batch_size or self.config.DEFAULT_VAL_BATCH
        
        # Validar dispositivo
        device = validate_device(device)
        
        # Construir comando de validação
        cmd = self._build_validation_command(
            weights=str(weights_path),
            img_size=img_size,
            device=device,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            name=name,
            batch_size=batch_size,
            verbose=verbose,
            **kwargs
        )
        
        # Imprimir configurações
        self._print_validation_info(
            weights=weights_path,
            img_size=img_size,
            device=device,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            name=name,
            batch_size=batch_size,
            verbose=verbose
        )
        
        # Executar validação
        logger.info("Iniciando validação...")
        result = subprocess.run(cmd, cwd=str(self.config.YOLOV9_DIR))
        
        if result.returncode == 0:
            results_dir = self.config.YOLOV9_DIR / "runs" / "val"
            logger.info(f"Validação concluída com sucesso! Resultados salvos em: {results_dir}")
            self._print_results_location()
        else:
            logger.error(f"Validação falhou com código: {result.returncode}")
        
        return result.returncode
    
    def _build_validation_command(
        self,
        weights: str,
        img_size: int,
        device: str,
        conf_threshold: float,
        iou_threshold: float,
        name: str,
        batch_size: int,
        verbose: bool,
        **kwargs
    ) -> list:
        """
        Constrói comando de validação.
        
        Args:
            weights: Caminho para pesos
            img_size: Tamanho da imagem
            device: Dispositivo
            conf_threshold: Threshold de confiança
            iou_threshold: Threshold de IoU
            name: Nome do experimento
            batch_size: Tamanho do batch
            verbose: Modo verboso
            **kwargs: Argumentos adicionais
        
        Returns:
            Lista com comando e argumentos
        """
        cmd = [
            sys.executable,
            str(self.config.VAL_SCRIPT),
            "--data", str(self.config.DATA_YAML),
            "--weights", weights,
            "--imgsz", str(img_size),
            "--device", device,
            "--conf-thres", str(conf_threshold),
            "--iou-thres", str(iou_threshold),
            "--name", name,
            "--batch-size", str(batch_size)
        ]
        
        if verbose:
            cmd.append("--verbose")
        
        # Adicionar argumentos extras
        for key, value in kwargs.items():
            if value is not None:
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        
        return cmd
    
    def _print_validation_info(
        self,
        weights: Path,
        img_size: int,
        device: str,
        conf_threshold: float,
        iou_threshold: float,
        name: str,
        batch_size: int,
        verbose: bool
    ) -> None:
        """
        Imprime informações da validação.
        
        Args:
            weights: Caminho para pesos
            img_size: Tamanho da imagem
            device: Dispositivo
            conf_threshold: Threshold de confiança
            iou_threshold: Threshold de IoU
            name: Nome do experimento
            batch_size: Tamanho do batch
            verbose: Modo verboso
        """
        print_header("VALIDAÇÃO YOLOv9 - Detector de Placas")
        
        print_config_summary(
            pesos=weights,
            img_size=img_size,
            device=device,
            confianca=conf_threshold,
            iou=iou_threshold,
            name=name,
            batch_size=batch_size,
            verbose="Sim" if verbose else "Não"
        )
        
        print("=" * 60)
    
    def _print_results_location(self) -> None:
        """Imprime localização dos resultados."""
        results_dir = self.config.YOLOV9_DIR / "runs" / "val"
        print(f"\nResultados salvos em: {results_dir}")


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Cria parser de argumentos para linha de comando.
    
    Returns:
        Parser configurado
    """
    parser = argparse.ArgumentParser(
        description="Validar modelo YOLOv9 de placas",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Caminho para os pesos do modelo (.pt)"
    )
    
    parser.add_argument(
        "--img-size",
        type=int,
        default=config.DEFAULT_IMG_SIZE,
        help="Tamanho da imagem"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device: 0 para GPU, cpu para CPU"
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=config.DEFAULT_VAL_CONF,
        help="Threshold de confiança"
    )
    
    parser.add_argument(
        "--iou",
        type=float,
        default=config.DEFAULT_VAL_IOU,
        help="Threshold de IoU"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default="validacao_placas",
        help="Nome do experimento"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.DEFAULT_VAL_BATCH,
        help="Tamanho do batch"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostrar métricas por classe"
    )
    
    return parser


def main() -> None:
    """Função principal para execução via linha de comando."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        validator = YOLOv9Validator()
        return_code = validator.validate(
            weights=args.weights,
            img_size=args.img_size,
            device=args.device,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            name=args.name,
            batch_size=args.batch_size,
            verbose=args.verbose
        )
        sys.exit(return_code)
        
    except Exception as e:
        logger.error(f"Erro durante validação: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()