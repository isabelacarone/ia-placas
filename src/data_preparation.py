"""
Módulo para preparação e organização dos dados do dataset.

Este módulo contém funções para descompactar, organizar e validar
os dados de treinamento, validação e teste.
"""

import zipfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

try:
    from .config import config
    from .utils import setup_logging, format_file_count
except ImportError:
    # Para execução standalone
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from config import config
    from utils import setup_logging, format_file_count


logger = setup_logging()


class DataPreparator:
    """Classe para preparação dos dados do dataset."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Inicializa o preparador de dados.
        
        Args:
            data_dir: Diretório de dados (usa config padrão se None)
        """
        self.data_dir = data_dir or config.DATA_DIR
        self.zip_mapping = {
            "Treino.zip": self.data_dir / "treino",
            "Validacao.zip": self.data_dir / "validacao", 
            "Teste.zip": self.data_dir / "teste"
        }
    
    def extract_datasets(self, zips_dir: Path) -> None:
        """
        Descompacta os arquivos ZIP do dataset.
        
        Args:
            zips_dir: Diretório contendo os arquivos ZIP
        
        Raises:
            FileNotFoundError: Se diretório de ZIPs não existir
        """
        if not zips_dir.exists():
            raise FileNotFoundError(f"Diretório não encontrado: {zips_dir}")
        
        logger.info(f"Iniciando extração de dados de: {zips_dir}")
        
        for zip_name, target_dir in self.zip_mapping.items():
            zip_path = zips_dir / zip_name
            
            if not zip_path.exists():
                logger.warning(f"Arquivo não encontrado: {zip_path}")
                continue
            
            self._extract_single_zip(zip_path, target_dir)
        
        logger.info("Extração de dados concluída!")
    
    def _extract_single_zip(self, zip_path: Path, target_dir: Path) -> None:
        """
        Extrai um único arquivo ZIP.
        
        Args:
            zip_path: Caminho para o arquivo ZIP
            target_dir: Diretório de destino
        """
        logger.info(f"Extraindo {zip_path.name} -> {target_dir}")
        
        # Criar diretório de destino
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Extrair arquivo
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            zip_file.extractall(target_dir)
        
        # Ajustar estrutura de pastas se necessário
        self._fix_directory_structure(target_dir)
        
        # Contar arquivos extraídos
        file_count = len(list(target_dir.rglob('*')))
        logger.info(f"  ✓ {file_count} arquivos extraídos")
    
    def _fix_directory_structure(self, target_dir: Path) -> None:
        """
        Ajusta estrutura de diretórios para compatibilidade com YOLO.
        
        Args:
            target_dir: Diretório para ajustar
        """
        # Renomear 'imagens' para 'images' se necessário
        images_dir = target_dir / "imagens"
        target_images_dir = target_dir / "images"
        
        if images_dir.exists() and not target_images_dir.exists():
            images_dir.rename(target_images_dir)
            logger.info(f"  ✓ Renomeado: {images_dir.name} -> {target_images_dir.name}")
    
    def validate_dataset_structure(self) -> Dict[str, Dict[str, int]]:
        """
        Valida a estrutura do dataset.
        
        Returns:
            Dicionário com estatísticas de cada split
        """
        logger.info("Validando estrutura do dataset...")
        
        statistics = {}
        
        for split_name in ["treino", "validacao", "teste"]:
            split_dir = self.data_dir / split_name
            stats = self._analyze_split_directory(split_dir, split_name)
            statistics[split_name] = stats
        
        self._print_dataset_summary(statistics)
        return statistics
    
    def _analyze_split_directory(self, split_dir: Path, split_name: str) -> Dict[str, int]:
        """
        Analisa um diretório de split do dataset.
        
        Args:
            split_dir: Diretório do split
            split_name: Nome do split
        
        Returns:
            Dicionário com estatísticas
        """
        if not split_dir.exists():
            logger.error(f"Diretório não encontrado: {split_dir}")
            return {"images": 0, "labels": 0, "errors": 1}
        
        # Contar arquivos
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        images = []
        for ext in image_extensions:
            images.extend(list(split_dir.rglob(f"*{ext}")))
            images.extend(list(split_dir.rglob(f"*{ext.upper()}")))
        
        labels = list(split_dir.rglob("*.txt"))
        
        # Verificar correspondência entre imagens e labels
        image_names = {img.stem for img in images}
        label_names = {lbl.stem for lbl in labels}
        
        missing_labels = image_names - label_names
        missing_images = label_names - image_names
        
        stats = {
            "images": len(images),
            "labels": len(labels),
            "missing_labels": len(missing_labels),
            "missing_images": len(missing_images),
            "errors": 0
        }
        
        # Log de avisos
        if missing_labels:
            logger.warning(f"{split_name}: {len(missing_labels)} imagens sem labels")
        if missing_images:
            logger.warning(f"{split_name}: {len(missing_images)} labels sem imagens")
        
        return stats
    
    def _print_dataset_summary(self, statistics: Dict[str, Dict[str, int]]) -> None:
        """
        Imprime resumo das estatísticas do dataset.
        
        Args:
            statistics: Estatísticas por split
        """
        print("\n" + "="*50)
        print("RESUMO DO DATASET")
        print("="*50)
        
        total_images = 0
        total_labels = 0
        
        for split_name, stats in statistics.items():
            print(f"\n{split_name.upper()}:")
            print(f"  Imagens: {stats['images']}")
            print(f"  Labels:  {stats['labels']}")
            
            if stats.get('missing_labels', 0) > 0:
                print(f"  ⚠️  Imagens sem labels: {stats['missing_labels']}")
            if stats.get('missing_images', 0) > 0:
                print(f"  ⚠️  Labels sem imagens: {stats['missing_images']}")
            
            total_images += stats['images']
            total_labels += stats['labels']
        
        print(f"\nTOTAL:")
        print(f"  Imagens: {total_images}")
        print(f"  Labels:  {total_labels}")
        print("="*50)
    
    def create_data_yaml(self, output_path: Optional[Path] = None) -> Path:
        """
        Cria arquivo YAML de configuração dos dados.
        
        Args:
            output_path: Caminho de saída (usa config padrão se None)
        
        Returns:
            Caminho do arquivo criado
        """
        if output_path is None:
            output_path = config.DATA_YAML
        
        yaml_content = f"""# Configuração do Dataset - Placas Veiculares
# Gerado automaticamente em {config.get_timestamp()}

# Caminho raiz do dataset (relativo ao YOLOv9)
path: ../../dados

# Caminhos dos splits (relativos ao path)
train: treino
val: validacao
test: teste

# Número de classes
nc: {config.NUM_CLASSES}

# Nomes das classes
names: {config.CLASS_NAMES}
"""
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        logger.info(f"Arquivo YAML criado: {output_path}")
        return output_path


def prepare_data_from_zips(zips_dir: str, data_dir: Optional[str] = None) -> None:
    """
    Função principal para preparar dados a partir de arquivos ZIP.
    
    Args:
        zips_dir: Diretório contendo os arquivos ZIP
        data_dir: Diretório de destino dos dados
    """
    zips_path = Path(zips_dir)
    data_path = Path(data_dir) if data_dir else config.DATA_DIR
    
    preparator = DataPreparator(data_path)
    preparator.extract_datasets(zips_path)
    preparator.validate_dataset_structure()
    preparator.create_data_yaml()