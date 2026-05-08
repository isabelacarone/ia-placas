# Referência da API - Detector de Placas YOLOv9

## Visão Geral

Esta documentação descreve a API dos módulos principais do projeto de detecção de placas veiculares.

## src.config

### ProjectConfig

Classe de configuração centralizada do projeto.

```python
from src.config import config

# Acessar configurações
print(config.NUM_CLASSES)  # 35
print(config.CLASS_NAMES)  # ['0', '1', ..., 'Z']
print(config.DEFAULT_EPOCHS)  # 300
```

#### Atributos Principais

| Atributo | Tipo | Descrição |
|----------|------|-----------|
| `ROOT_DIR` | Path | Diretório raiz do projeto |
| `DATA_DIR` | Path | Diretório dos dados |
| `YOLOV9_DIR` | Path | Diretório do YOLOv9 |
| `NUM_CLASSES` | int | Número de classes (35) |
| `CLASS_NAMES` | List[str] | Nomes das classes |
| `DEFAULT_EPOCHS` | int | Épocas padrão (300) |
| `DEFAULT_BATCH_SIZE` | int | Batch size padrão (8) |

#### Métodos

##### create_directories()
```python
config.create_directories()
```
Cria todos os diretórios necessários do projeto.

##### validate_paths()
```python
validation_result = config.validate_paths()
# Returns: Dict[str, bool]
```
Valida se os caminhos essenciais existem.

## src.utils

### Funções Utilitárias

#### setup_logging()
```python
from src.utils import setup_logging

logger = setup_logging(
    level=logging.INFO,
    log_file=Path("logs/app.log"),
    format_string="%(asctime)s - %(levelname)s - %(message)s"
)
```

**Parâmetros:**
- `level` (int): Nível de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `log_file` (Path, opcional): Arquivo para salvar logs
- `format_string` (str, opcional): Formato das mensagens

**Retorna:** `logging.Logger`

#### validate_device()
```python
from src.utils import validate_device

device = validate_device("0")      # "0"
device = validate_device("cpu")    # "cpu"
device = validate_device("cuda:1") # "1"
```

**Parâmetros:**
- `device` (str): Especificação do dispositivo

**Retorna:** `str` - Dispositivo normalizado

**Raises:** `ValueError` - Se dispositivo inválido

#### check_yolov9_installation()
```python
from src.utils import check_yolov9_installation

is_installed = check_yolov9_installation()
# Returns: bool
```

Verifica se o YOLOv9 está instalado corretamente.

#### parse_yolo_label()
```python
from src.utils import parse_yolo_label

detections = parse_yolo_label(Path("label.txt"))
# Returns: List[Dict[str, Union[int, float]]]
```

Parseia arquivo de label do YOLO.

## src.data_preparation

### DataPreparator

Classe para preparação e organização dos dados.

```python
from src.data_preparation import DataPreparator

preparator = DataPreparator(data_dir=Path("dados"))
```

#### Métodos

##### extract_datasets()
```python
preparator.extract_datasets(zips_dir=Path("zips/"))
```

**Parâmetros:**
- `zips_dir` (Path): Diretório contendo os arquivos ZIP

**Raises:**
- `FileNotFoundError`: Se diretório não existir

##### validate_dataset_structure()
```python
statistics = preparator.validate_dataset_structure()
# Returns: Dict[str, Dict[str, int]]
```

Valida estrutura do dataset e retorna estatísticas.

**Retorna:**
```python
{
    "treino": {
        "images": 1000,
        "labels": 1000,
        "missing_labels": 0,
        "missing_images": 0,
        "errors": 0
    },
    # ... outros splits
}
```

##### create_data_yaml()
```python
yaml_path = preparator.create_data_yaml(output_path=Path("config.yaml"))
# Returns: Path
```

Cria arquivo YAML de configuração dos dados.

### Função de Conveniência

#### prepare_data_from_zips()
```python
from src.data_preparation import prepare_data_from_zips

prepare_data_from_zips(
    zips_dir="caminho/para/zips",
    data_dir="dados"  # opcional
)
```

Função principal para preparar dados a partir de ZIPs.

## src.training

### YOLOv9Trainer

Classe para treinamento do modelo YOLOv9.

```python
from src.training import YOLOv9Trainer

trainer = YOLOv9Trainer()
```

#### Métodos

##### train()
```python
return_code = trainer.train(
    epochs=300,
    batch_size=8,
    img_size=640,
    device="0",
    workers=4,
    name="detector_placas",
    resume=None,
    weights="",
    **kwargs
)
# Returns: int (código de retorno)
```

**Parâmetros:**
- `epochs` (int, opcional): Número de épocas
- `batch_size` (int, opcional): Tamanho do batch
- `img_size` (int, opcional): Tamanho da imagem
- `device` (str): Dispositivo ('0' para GPU, 'cpu' para CPU)
- `workers` (int, opcional): Número de workers
- `name` (str): Nome do experimento
- `resume` (str, opcional): Caminho para checkpoint
- `weights` (str): Pesos pré-treinados
- `**kwargs`: Argumentos adicionais

**Retorna:** `int` - Código de retorno (0 = sucesso)

**Raises:**
- `ValueError`: Parâmetros inválidos
- `RuntimeError`: Erro durante treinamento
- `FileNotFoundError`: Arquivos não encontrados

## src.validation

### YOLOv9Validator

Classe para validação do modelo YOLOv9.

```python
from src.validation import YOLOv9Validator

validator = YOLOv9Validator()
```

#### Métodos

##### validate()
```python
return_code = validator.validate(
    weights="pesos/best.pt",
    img_size=640,
    device="0",
    conf_threshold=0.001,
    iou_threshold=0.7,
    name="validacao_placas",
    batch_size=32,
    verbose=False,
    **kwargs
)
# Returns: int
```

**Parâmetros:**
- `weights` (str): Caminho para os pesos do modelo
- `img_size` (int, opcional): Tamanho da imagem
- `device` (str): Dispositivo
- `conf_threshold` (float, opcional): Threshold de confiança
- `iou_threshold` (float, opcional): Threshold de IoU
- `name` (str): Nome do experimento
- `batch_size` (int, opcional): Tamanho do batch
- `verbose` (bool): Mostrar métricas por classe
- `**kwargs`: Argumentos adicionais

## src.detection

### YOLOv9Detector

Classe para detecção com modelo YOLOv9.

```python
from src.detection import YOLOv9Detector

detector = YOLOv9Detector()
```

#### Métodos

##### detect()
```python
return_code = detector.detect(
    weights="pesos/best.pt",
    source="imagens/",
    img_size=640,
    device="0",
    conf_threshold=0.25,
    iou_threshold=0.45,
    name="deteccao_placas",
    save_txt=False,
    save_conf=False,
    **kwargs
)
# Returns: int
```

**Parâmetros:**
- `weights` (str): Caminho para os pesos do modelo
- `source` (str): Imagem, pasta ou vídeo
- `img_size` (int, opcional): Tamanho da imagem
- `device` (str): Dispositivo
- `conf_threshold` (float, opcional): Threshold de confiança
- `iou_threshold` (float, opcional): Threshold de IoU NMS
- `name` (str): Nome do experimento
- `save_txt` (bool): Salvar resultados em .txt
- `save_conf` (bool): Salvar confianças nos .txt
- `**kwargs`: Argumentos adicionais

### PlateCharacterProcessor

Classe para processamento de caracteres detectados.

```python
from src.detection import PlateCharacterProcessor

processor = PlateCharacterProcessor()
```

#### Métodos

##### process_detection_results()
```python
results = processor.process_detection_results(
    results_dir=Path("runs/detect/exp/"),
    confidence_threshold=0.5
)
# Returns: Dict[str, List[Dict[str, Any]]]
```

**Parâmetros:**
- `results_dir` (Path): Diretório com resultados da detecção
- `confidence_threshold` (float): Threshold mínimo de confiança

**Retorna:**
```python
{
    "imagem1": {
        "detections": [
            {
                "class_id": 0,
                "class_name": "A",
                "x_center": 0.5,
                "y_center": 0.3,
                "width": 0.1,
                "height": 0.2,
                "confidence": 0.95
            },
            # ... outras detecções
        ],
        "plate_text": "ABC1234",
        "confidence": 0.87
    },
    # ... outras imagens
}
```

## Exemplos de Uso Completos

### Preparação de Dados
```python
from src.data_preparation import DataPreparator
from pathlib import Path

# Inicializar preparador
preparator = DataPreparator()

# Extrair dados
preparator.extract_datasets(Path("zips/"))

# Validar estrutura
stats = preparator.validate_dataset_structure()
print(f"Total de imagens: {sum(s['images'] for s in stats.values())}")

# Criar configuração YAML
yaml_path = preparator.create_data_yaml()
```

### Treinamento Completo
```python
from src.training import YOLOv9Trainer
from src.utils import setup_logging

# Configurar logging
logger = setup_logging()

# Inicializar treinador
trainer = YOLOv9Trainer()

# Treinar modelo
return_code = trainer.train(
    epochs=100,
    batch_size=16,
    device="0",
    name="meu_detector"
)

if return_code == 0:
    logger.info("Treinamento concluído com sucesso!")
```

### Detecção e Processamento
```python
from src.detection import YOLOv9Detector, PlateCharacterProcessor
from pathlib import Path

# Executar detecção
detector = YOLOv9Detector()
detector.detect(
    weights="pesos/best.pt",
    source="imagens_teste/",
    save_txt=True,
    save_conf=True
)

# Processar resultados
processor = PlateCharacterProcessor()
results = processor.process_detection_results(
    results_dir=Path("uvv/yolov9-main/runs/detect/deteccao_placas/"),
    confidence_threshold=0.6
)

# Exibir placas detectadas
for image_name, data in results.items():
    print(f"{image_name}: {data['plate_text']} (conf: {data['confidence']:.2f})")
```

### Pipeline Completo
```python
from src.data_preparation import prepare_data_from_zips
from src.training import YOLOv9Trainer
from src.validation import YOLOv9Validator
from src.detection import YOLOv9Detector

# 1. Preparar dados
prepare_data_from_zips("zips/")

# 2. Treinar modelo
trainer = YOLOv9Trainer()
trainer.train(epochs=200, batch_size=8)

# 3. Validar modelo
validator = YOLOv9Validator()
validator.validate(
    weights="uvv/yolov9-main/runs/train/detector_placas/weights/best.pt",
    verbose=True
)

# 4. Executar detecção
detector = YOLOv9Detector()
detector.detect(
    weights="uvv/yolov9-main/runs/train/detector_placas/weights/best.pt",
    source="imagens_teste/"
)
```

## Tratamento de Exceções

### Exceções Comuns

```python
from src.training import YOLOv9Trainer
from src.utils import setup_logging

logger = setup_logging()

try:
    trainer = YOLOv9Trainer()
    trainer.train(epochs=100)
    
except FileNotFoundError as e:
    logger.error(f"Arquivo não encontrado: {e}")
    
except ValueError as e:
    logger.error(f"Parâmetro inválido: {e}")
    
except RuntimeError as e:
    logger.error(f"Erro de execução: {e}")
    
except Exception as e:
    logger.error(f"Erro inesperado: {e}")
```

## Configuração de Logging

### Logging Básico
```python
from src.utils import setup_logging
import logging

# Logging simples
logger = setup_logging()

# Logging com arquivo
logger = setup_logging(
    level=logging.DEBUG,
    log_file=Path("logs/debug.log")
)

# Logging customizado
logger = setup_logging(
    level=logging.INFO,
    format_string="[%(levelname)s] %(asctime)s - %(message)s"
)
```

### Uso do Logger
```python
logger.debug("Informação de debug")
logger.info("Processo iniciado")
logger.warning("Aviso importante")
logger.error("Erro ocorreu")
logger.critical("Erro crítico")
```