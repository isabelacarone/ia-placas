"""
Módulo de detecção/inferência do modelo YOLOv9.

Este módulo contém classes e funções para executar detecção
de caracteres em placas usando o modelo treinado.
"""

import re
import sys
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
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


class YOLOv9Detector:
    """Classe para detecção com modelo YOLOv9."""
    
    def __init__(self):
        """Inicializa o detector."""
        self.config = config
        self._validate_setup()
    
    def _validate_setup(self) -> None:
        """Valida se o ambiente está configurado corretamente."""
        if not check_yolov9_installation():
            raise RuntimeError(
                "YOLOv9 não está instalado corretamente. "
                "Verifique se a pasta uvv/yolov9-main/ existe."
            )
    
    def detect(
        self,
        weights: str,
        source: str,
        img_size: int = None,
        device: str = "0",
        conf_threshold: float = None,
        iou_threshold: float = None,
        name: str = "deteccao_placas",
        save_txt: bool = False,
        save_conf: bool = False,
        **kwargs
    ) -> int:
        """
        Executa detecção em imagens.
        
        Args:
            weights: Caminho para os pesos do modelo (.pt)
            source: Imagem, pasta de imagens, ou vídeo
            img_size: Tamanho da imagem (padrão: config.DEFAULT_IMG_SIZE)
            device: Dispositivo ('0' para GPU, 'cpu' para CPU)
            conf_threshold: Threshold de confiança (padrão: config.DEFAULT_CONF_THRESHOLD)
            iou_threshold: Threshold de IoU NMS (padrão: config.DEFAULT_IOU_THRESHOLD)
            name: Nome do experimento
            save_txt: Salvar resultados em .txt
            save_conf: Salvar confianças nos .txt
            **kwargs: Argumentos adicionais
        
        Returns:
            Código de retorno do processo de detecção
        
        Raises:
            FileNotFoundError: Se pesos ou source não encontrados
            ValueError: Se parâmetros inválidos
        """
        # Validar arquivos
        weights_path = Path(weights)
        if not weights_path.exists():
            raise FileNotFoundError(f"Pesos não encontrados: {weights_path}")
        
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Source não encontrado: {source_path}")
        
        # Usar valores padrão se não especificados
        img_size = img_size or self.config.DEFAULT_IMG_SIZE
        conf_threshold = conf_threshold or self.config.DEFAULT_CONF_THRESHOLD
        iou_threshold = iou_threshold or self.config.DEFAULT_IOU_THRESHOLD
        
        # Validar dispositivo
        device = validate_device(device)
        
        # Construir comando de detecção
        cmd = self._build_detection_command(
            weights=str(weights_path),
            source=str(source_path),
            img_size=img_size,
            device=device,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            name=name,
            save_txt=save_txt,
            save_conf=save_conf,
            **kwargs
        )
        
        # Imprimir configurações
        self._print_detection_info(
            weights=weights_path,
            source=source_path,
            img_size=img_size,
            device=device,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            name=name,
            save_txt=save_txt,
            save_conf=save_conf
        )
        
        # Executar detecção
        logger.info("Iniciando detecção...")
        result = subprocess.run(cmd, cwd=str(self.config.YOLOV9_DIR))
        
        if result.returncode == 0:
            results_dir = self.config.YOLOV9_DIR / "runs" / "detect"
            logger.info(f"Detecção concluída com sucesso! Resultados salvos em: {results_dir}")
            self._print_results_location()
        else:
            logger.error(f"Detecção falhou com código: {result.returncode}")
        
        return result.returncode
    
    def _build_detection_command(
        self,
        weights: str,
        source: str,
        img_size: int,
        device: str,
        conf_threshold: float,
        iou_threshold: float,
        name: str,
        save_txt: bool,
        save_conf: bool,
        **kwargs
    ) -> List[str]:
        """
        Constrói comando de detecção.
        
        Args:
            weights: Caminho para pesos
            source: Source para detecção
            img_size: Tamanho da imagem
            device: Dispositivo
            conf_threshold: Threshold de confiança
            iou_threshold: Threshold de IoU
            name: Nome do experimento
            save_txt: Salvar em texto
            save_conf: Salvar confiança
            **kwargs: Argumentos adicionais
        
        Returns:
            Lista com comando e argumentos
        """
        cmd = [
            sys.executable,
            str(self.config.DETECT_SCRIPT),
            "--weights", weights,
            "--source", source,
            "--img-size", str(img_size),
            "--device", device,
            "--conf-thres", str(conf_threshold),
            "--iou-thres", str(iou_threshold),
            "--name", name
        ]
        
        if save_txt:
            cmd.append("--save-txt")
        
        if save_conf:
            cmd.append("--save-conf")
        
        # Adicionar argumentos extras
        for key, value in kwargs.items():
            if value is not None:
                if isinstance(value, bool) and value:
                    cmd.append(f"--{key.replace('_', '-')}")
                else:
                    cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        
        return cmd
    
    def _print_detection_info(
        self,
        weights: Path,
        source: Path,
        img_size: int,
        device: str,
        conf_threshold: float,
        iou_threshold: float,
        name: str,
        save_txt: bool,
        save_conf: bool
    ) -> None:
        """
        Imprime informações da detecção.
        
        Args:
            weights: Caminho para pesos
            source: Source para detecção
            img_size: Tamanho da imagem
            device: Dispositivo
            conf_threshold: Threshold de confiança
            iou_threshold: Threshold de IoU
            name: Nome do experimento
            save_txt: Salvar em texto
            save_conf: Salvar confiança
        """
        print_header("DETECÇÃO YOLOv9 - Detector de Placas")
        
        print_config_summary(
            pesos=weights,
            source=source,
            img_size=img_size,
            device=device,
            confianca=conf_threshold,
            iou=iou_threshold,
            name=name,
            salvar_txt="Sim" if save_txt else "Não",
            salvar_conf="Sim" if save_conf else "Não"
        )
        
        print("=" * 60)
    
    def _print_results_location(self) -> None:
        """Imprime localização dos resultados."""
        results_dir = self.config.YOLOV9_DIR / "runs" / "detect"
        print(f"\nResultados salvos em: {results_dir}")


class PlateCharacterProcessor:
    """Classe para processamento de caracteres detectados em placas."""
    
    def __init__(self):
        """Inicializa o processador."""
        self.class_mapping = {i: name for i, name in enumerate(config.CLASS_NAMES)}
    
    def process_detection_results(
        self,
        results_dir: Path,
        confidence_threshold: float = 0.5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Processa resultados de detecção para reconstruir placas.
        
        Args:
            results_dir: Diretório com resultados da detecção
            confidence_threshold: Threshold mínimo de confiança
        
        Returns:
            Dicionário com placas processadas por imagem
        """
        processed_results = {}
        
        # Buscar arquivos de resultado
        label_files = list(results_dir.rglob("*.txt"))
        
        for label_file in label_files:
            image_name = label_file.stem
            detections = self._parse_detection_file(label_file, confidence_threshold)
            
            if detections:
                plate_text = self._reconstruct_plate_text(detections)
                processed_results[image_name] = {
                    'detections': detections,
                    'plate_text': plate_text,
                    'confidence': self._calculate_average_confidence(detections)
                }
        
        return processed_results
    
    def _filter_by_confidence(
        self,
        detections: List[Dict],
        threshold: float
    ) -> List[Dict]:
        """
        Filtra detecções pelo valor de confiança.
        
        Args:
            detections: Lista de detecções a filtrar
            threshold: Threshold mínimo de confiança (inclusive)
        
        Returns:
            Lista contendo apenas detecções com confidence >= threshold
        """
        return [d for d in detections if d['confidence'] >= threshold]

    def _parse_detection_file(
        self,
        label_file: Path,
        confidence_threshold: float
    ) -> List[Dict[str, Any]]:
        """
        Parseia arquivo de detecção.
        
        Args:
            label_file: Arquivo com detecções
            confidence_threshold: Threshold de confiança
        
        Returns:
            Lista de detecções válidas
        """
        raw_detections = []
        
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        confidence = float(parts[5]) if len(parts) > 5 else 1.0
                        
                        detection = {
                            'class_id': class_id,
                            'class_name': self.class_mapping.get(class_id, 'unknown'),
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': width,
                            'height': height,
                            'confidence': confidence
                        }
                        raw_detections.append(detection)
        
        except Exception as e:
            logger.warning(f"Erro ao processar {label_file}: {e}")
        
        return self._filter_by_confidence(raw_detections, confidence_threshold)
    
    def _reconstruct_plate_text(self, detections: List[Dict[str, Any]]) -> str:
        """
        Reconstrói texto da placa a partir das detecções.
        
        Args:
            detections: Lista de detecções
        
        Returns:
            Texto reconstruído da placa
        """
        # Ordenar detecções por posição (esquerda para direita)
        sorted_detections = sorted(detections, key=lambda x: x['x_center'])
        
        # Extrair caracteres
        characters = [det['class_name'] for det in sorted_detections]
        
        return ''.join(characters)
    
    def _calculate_average_confidence(self, detections: List[Dict[str, Any]]) -> float:
        """
        Calcula confiança média das detecções.
        
        Args:
            detections: Lista de detecções
        
        Returns:
            Confiança média
        """
        if not detections:
            return 0.0
        
        confidences = [det['confidence'] for det in detections]
        return sum(confidences) / len(confidences)

    def _validate_plate_format(self, plate_text: str) -> Dict[str, Any]:
        """
        Valida se o texto da placa corresponde a um formato brasileiro válido.

        Formatos suportados:
        - Placa_Antiga:   3 letras + 4 dígitos  (ex: ABC1234)
        - Placa_Mercosul: 3 letras + 1 dígito + 1 letra + 2 dígitos (ex: ABC1D23)

        Args:
            plate_text: Texto reconstruído da placa

        Returns:
            Dicionário com 'valid_format' (bool) e 'plate_type' (str ou None)
        """
        # Comprimento diferente de 7 → inválido imediatamente
        if len(plate_text) != 7:
            return {'valid_format': False, 'plate_type': None}

        # Padrão Placa_Antiga: AAA0000
        if re.match(r'^[A-Z]{3}[0-9]{4}$', plate_text):
            return {'valid_format': True, 'plate_type': 'Placa_Antiga'}

        # Padrão Placa_Mercosul: AAA0A00
        if re.match(r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$', plate_text):
            return {'valid_format': True, 'plate_type': 'Placa_Mercosul'}

        # Nenhum padrão reconhecido
        return {'valid_format': False, 'plate_type': None}


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Cria parser de argumentos para linha de comando.
    
    Returns:
        Parser configurado
    """
    parser = argparse.ArgumentParser(
        description="Detectar caracteres em placas com YOLOv9",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Caminho para os pesos do modelo (.pt)"
    )
    
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Imagem, pasta de imagens, ou vídeo"
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
        default=config.DEFAULT_CONF_THRESHOLD,
        help="Threshold de confiança"
    )
    
    parser.add_argument(
        "--iou",
        type=float,
        default=config.DEFAULT_IOU_THRESHOLD,
        help="Threshold de IoU NMS"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default="deteccao_placas",
        help="Nome do experimento"
    )
    
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Salvar resultados em .txt"
    )
    
    parser.add_argument(
        "--save-conf",
        action="store_true",
        help="Salvar confianças nos .txt"
    )
    
    return parser


def main() -> None:
    """Função principal para execução via linha de comando."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        detector = YOLOv9Detector()
        return_code = detector.detect(
            weights=args.weights,
            source=args.source,
            img_size=args.img_size,
            device=args.device,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            name=args.name,
            save_txt=args.save_txt,
            save_conf=args.save_conf
        )
        sys.exit(return_code)
        
    except Exception as e:
        logger.error(f"Erro durante detecção: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()