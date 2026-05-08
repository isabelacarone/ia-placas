"""
Módulo de treinamento do modelo YOLOv9.

Este módulo contém classes e funções para treinar o modelo YOLOv9
para detecção de caracteres em placas veiculares.
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


class YOLOv9Trainer:
    """Classe para treinamento do modelo YOLOv9."""
    
    def __init__(self):
        """Inicializa o treinador."""
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
    
    def train(
        self,
        epochs: int = None,
        batch_size: int = None,
        img_size: int = None,
        device: str = "0",
        workers: int = None,
        name: str = "detector_placas",
        resume: Optional[str] = None,
        weights: str = "",
        **kwargs
    ) -> int:
        """
        Executa o treinamento do modelo.
        
        Args:
            epochs: Número de épocas (padrão: config.DEFAULT_EPOCHS)
            batch_size: Tamanho do batch (padrão: config.DEFAULT_BATCH_SIZE)
            img_size: Tamanho da imagem (padrão: config.DEFAULT_IMG_SIZE)
            device: Dispositivo ('0' para GPU, 'cpu' para CPU)
            workers: Número de workers (padrão: config.DEFAULT_WORKERS)
            name: Nome do experimento
            resume: Caminho para checkpoint para retomar treino
            weights: Pesos pré-treinados (vazio = treinar do zero)
            **kwargs: Argumentos adicionais
        
        Returns:
            Código de retorno do processo de treinamento
        
        Raises:
            ValueError: Se parâmetros inválidos
            RuntimeError: Se erro durante treinamento
        """
        # Usar valores padrão se não especificados
        epochs = epochs or self.config.DEFAULT_EPOCHS
        batch_size = batch_size or self.config.DEFAULT_BATCH_SIZE
        img_size = img_size or self.config.DEFAULT_IMG_SIZE
        workers = workers or self.config.DEFAULT_WORKERS
        
        # Validar dispositivo
        device = validate_device(device)
        
        # Construir comando de treinamento
        cmd = self._build_train_command(
            epochs=epochs,
            batch_size=batch_size,
            img_size=img_size,
            device=device,
            workers=workers,
            name=name,
            resume=resume,
            weights=weights,
            **kwargs
        )
        
        # Imprimir configurações
        self._print_training_info(
            epochs=epochs,
            batch_size=batch_size,
            img_size=img_size,
            device=device,
            name=name,
            resume=resume,
            weights=weights
        )
        
        # Executar treinamento
        logger.info("Iniciando treinamento...")
        result = subprocess.run(cmd, cwd=str(self.config.YOLOV9_DIR))
        
        if result.returncode == 0:
            logger.info("Treinamento concluído com sucesso!")
        else:
            logger.error(f"Treinamento falhou com código: {result.returncode}")
        
        return result.returncode
    
    def _build_train_command(
        self,
        epochs: int,
        batch_size: int,
        img_size: int,
        device: str,
        workers: int,
        name: str,
        resume: Optional[str],
        weights: str,
        **kwargs
    ) -> list:
        """
        Constrói comando de treinamento.
        
        Args:
            epochs: Número de épocas
            batch_size: Tamanho do batch
            img_size: Tamanho da imagem
            device: Dispositivo
            workers: Número de workers
            name: Nome do experimento
            resume: Checkpoint para retomar
            weights: Pesos pré-treinados
            **kwargs: Argumentos adicionais
        
        Returns:
            Lista com comando e argumentos
        """
        cmd = [
            sys.executable,
            str(self.config.TRAIN_SCRIPT),
            "--workers", str(workers),
            "--device", device,
            "--batch-size", str(batch_size),
            "--data", str(self.config.DATA_YAML),
            "--img", str(img_size),
            "--cfg", str(self.config.MODEL_YAML),
            "--weights", weights,
            "--name", name,
            "--hyp", str(self.config.HYPERPARAMS_YAML),
            "--epochs", str(epochs),
            "--noplots"
        ]
        
        if resume:
            cmd.extend(["--resume", resume])
        
        # Adicionar argumentos extras
        for key, value in kwargs.items():
            if value is not None:
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        
        return cmd
    
    def _print_training_info(
        self,
        epochs: int,
        batch_size: int,
        img_size: int,
        device: str,
        name: str,
        resume: Optional[str],
        weights: str
    ) -> None:
        """
        Imprime informações do treinamento.
        
        Args:
            epochs: Número de épocas
            batch_size: Tamanho do batch
            img_size: Tamanho da imagem
            device: Dispositivo
            name: Nome do experimento
            resume: Checkpoint para retomar
            weights: Pesos pré-treinados
        """
        print_header("TREINAMENTO YOLOv9 - Detector de Placas")
        
        print_config_summary(
            epochs=epochs,
            batch_size=batch_size,
            img_size=img_size,
            device=device,
            name=name,
            data_yaml=self.config.DATA_YAML,
            model_yaml=self.config.MODEL_YAML,
            hyperparams=self.config.HYPERPARAMS_YAML
        )
        
        if resume:
            print(f"  Retomar de: {resume}")
        if weights:
            print(f"  Pesos iniciais: {weights}")
        
        print("=" * 60)


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Cria parser de argumentos para linha de comando.
    
    Returns:
        Parser configurado
    """
    parser = argparse.ArgumentParser(
        description="Treinar YOLOv9 para detecção de placas",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=config.DEFAULT_EPOCHS,
        help="Número de épocas"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=config.DEFAULT_BATCH_SIZE,
        help="Tamanho do batch"
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
        "--workers", 
        type=int, 
        default=config.DEFAULT_WORKERS,
        help="Número de workers para dataloader"
    )
    
    parser.add_argument(
        "--name", 
        type=str, 
        default="detector_placas",
        help="Nome do experimento"
    )
    
    parser.add_argument(
        "--resume", 
        type=str, 
        default="",
        help="Caminho para checkpoint para retomar treino"
    )
    
    parser.add_argument(
        "--weights", 
        type=str, 
        default="",
        help="Pesos pré-treinados (vazio = treinar do zero)"
    )
    
    return parser


def main() -> None:
    """Função principal para execução via linha de comando."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        trainer = YOLOv9Trainer()
        return_code = trainer.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            device=args.device,
            workers=args.workers,
            name=args.name,
            resume=args.resume if args.resume else None,
            weights=args.weights
        )
        sys.exit(return_code)
        
    except Exception as e:
        logger.error(f"Erro durante treinamento: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()