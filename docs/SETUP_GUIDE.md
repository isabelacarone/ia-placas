# Guia de Configuração - Detector de Placas YOLOv9

## Pré-requisitos

### Sistema Operacional
- Linux (Ubuntu 18.04+, CentOS 7+)
- Windows 10/11
- macOS 10.15+

### Hardware Recomendado
- **GPU**: NVIDIA com CUDA 11.8+ e 8GB+ VRAM
- **RAM**: 16GB+ (32GB recomendado para datasets grandes)
- **Storage**: 50GB+ espaço livre (SSD recomendado)
- **CPU**: 8+ cores para processamento paralelo

### Software
- Python 3.8+
- Git
- CUDA Toolkit 11.8+ (para GPU)

## Instalação

### 1. Clonar o Repositório

```bash
git clone <url-do-repositorio>
cd detector-placas-yolov9
```

### 2. Criar Ambiente Virtual

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente virtual
# Linux/macOS:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### 3. Instalar PyTorch

#### Para GPU (CUDA 12.1)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### Para GPU (CUDA 11.8)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Para CPU apenas
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 4. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 5. Baixar YOLOv9

```bash
# Criar diretório
mkdir -p uvv

# Clonar YOLOv9
cd uvv
git clone https://github.com/WongKinYiu/yolov9.git yolov9-main
cd ..
```

### 6. Verificar Instalação

```bash
python -c "from src.utils import check_yolov9_installation; print('✓ YOLOv9 OK' if check_yolov9_installation() else '✗ YOLOv9 Erro')"
python -c "import torch; print('✓ CUDA OK' if torch.cuda.is_available() else '✗ CUDA Indisponível')"
```

## Configuração do Dataset

### Estrutura Esperada

```
dados/
├── treino/
│   ├── images/          # Imagens de treinamento (.jpg, .png)
│   └── labels/          # Labels YOLO (.txt)
├── validacao/
│   ├── images/          # Imagens de validação
│   └── labels/          # Labels de validação
└── teste/
    ├── images/          # Imagens de teste
    └── labels/          # Labels de teste
```

### Opção 1: Usar ZIPs Existentes

Se você tem arquivos `Treino.zip`, `Validacao.zip` e `Teste.zip`:

```bash
python scripts/prepare_data.py --zips-dir "caminho/para/os/zips"
```

### Opção 2: Organizar Manualmente

1. Criar estrutura de diretórios:
```bash
mkdir -p dados/{treino,validacao,teste}/{images,labels}
```

2. Copiar imagens e labels para as pastas correspondentes

3. Validar estrutura:
```bash
python -c "
from src.data_preparation import DataPreparator
prep = DataPreparator()
stats = prep.validate_dataset_structure()
print('Dataset validado!')
"
```

## Configuração de Treinamento

### Arquivo de Configuração (configs/dados-placas.yaml)

```yaml
# Caminho raiz do dataset (relativo ao YOLOv9)
path: ../../dados

# Caminhos dos splits (relativos ao path)
train: treino
val: validacao
test: teste

# Número de classes
nc: 35

# Nomes das classes (dígitos 0-9 e letras A-Z, exceto O)
names: [
  '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
  'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U',
  'V', 'W', 'X', 'Y', 'Z'
]
```

### Hiperparâmetros (configs/hyp.scratch-high.yaml)

```yaml
# Learning rate
lr0: 0.01                    # Taxa de aprendizado inicial
lrf: 0.01                    # Taxa de aprendizado final

# Otimização
momentum: 0.937              # Momentum do SGD
weight_decay: 0.0005         # Decaimento de peso

# Warmup
warmup_epochs: 3.0           # Épocas de warmup
warmup_momentum: 0.8         # Momentum durante warmup
warmup_bias_lr: 0.1          # LR de bias durante warmup

# Loss weights
box: 7.5                     # Peso da loss de bounding box
cls: 0.5                     # Peso da loss de classificação
obj: 0.7                     # Peso da loss de objetividade
dfl: 1.5                     # Peso da loss DFL

# Anchors
iou_t: 0.20                  # Threshold de IoU para anchors
anchor_t: 5.0                # Threshold de anchor

# Data augmentation
hsv_h: 0.015                 # Augmentation de matiz (0-1)
hsv_s: 0.7                   # Augmentation de saturação (0-1)
hsv_v: 0.4                   # Augmentation de valor (0-1)
degrees: 0.0                 # Rotação (graus)
translate: 0.1               # Translação (fração)
scale: 0.9                   # Escala (ganho)
shear: 0.0                   # Cisalhamento (graus)
perspective: 0.0             # Perspectiva (0-0.001)
flipud: 0.0                  # Flip vertical (probabilidade)
fliplr: 0.0                  # Flip horizontal (probabilidade)
mosaic: 1.0                  # Probabilidade de mosaic
mixup: 0.15                  # Probabilidade de mixup
copy_paste: 0.3              # Probabilidade de copy-paste
```

## Uso Básico

### 1. Preparar Dados

```bash
# Se você tem ZIPs
python scripts/prepare_data.py --zips-dir "/caminho/para/zips"

# Verificar estrutura
python -c "
from src.data_preparation import DataPreparator
prep = DataPreparator()
prep.validate_dataset_structure()
"
```

### 2. Treinar Modelo

```bash
# Treinamento básico
python scripts/train.py

# Treinamento customizado
python scripts/train.py \
    --epochs 300 \
    --batch-size 16 \
    --device 0 \
    --name meu_detector

# Retomar treinamento
python scripts/train.py \
    --resume uvv/yolov9-main/runs/train/detector_placas/weights/last.pt
```

### 3. Validar Modelo

```bash
# Validação básica
python scripts/validate.py \
    --weights uvv/yolov9-main/runs/train/detector_placas/weights/best.pt

# Validação detalhada
python scripts/validate.py \
    --weights pesos/best.pt \
    --verbose \
    --device cpu
```

### 4. Executar Detecção

```bash
# Detecção em pasta
python scripts/detect.py \
    --weights pesos/best.pt \
    --source imagens/

# Detecção em imagem específica
python scripts/detect.py \
    --weights pesos/best.pt \
    --source imagem.jpg \
    --conf 0.25 \
    --save-txt

# Detecção com configurações customizadas
python scripts/detect.py \
    --weights pesos/best.pt \
    --source imagens/ \
    --conf 0.5 \
    --iou 0.4 \
    --device cpu \
    --save-txt \
    --save-conf
```

## Configurações Avançadas

### Otimização de Performance

#### Para GPUs com Pouca VRAM (4-6GB)
```bash
python scripts/train.py \
    --batch-size 4 \
    --workers 2 \
    --img-size 416
```

#### Para GPUs Potentes (12GB+)
```bash
python scripts/train.py \
    --batch-size 32 \
    --workers 8 \
    --img-size 640
```

#### Para CPU (não recomendado para treinamento)
```bash
python scripts/train.py \
    --device cpu \
    --batch-size 2 \
    --workers 0
```

### Configuração de Logging

```python
# Criar arquivo de configuração de logging
# logs/logging_config.py

import logging
from pathlib import Path

def setup_project_logging():
    """Configura logging para todo o projeto."""
    
    # Criar diretório de logs
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configurar formatador
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Logger principal
    logger = logging.getLogger("placas_detector")
    logger.setLevel(logging.INFO)
    
    # Handler para arquivo
    file_handler = logging.FileHandler(log_dir / "detector.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Handler para console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger
```

### Monitoramento com TensorBoard

```bash
# Instalar TensorBoard (se não instalado)
pip install tensorboard

# Visualizar logs de treinamento
tensorboard --logdir uvv/yolov9-main/runs/train

# Acessar no navegador: http://localhost:6006
```

## Troubleshooting

### Problemas Comuns

#### 1. CUDA Out of Memory
```bash
# Soluções:
# - Reduzir batch_size
python scripts/train.py --batch-size 4

# - Reduzir tamanho da imagem
python scripts/train.py --img-size 416

# - Usar gradient accumulation (se disponível)
python scripts/train.py --batch-size 4 --accumulate 4
```

#### 2. Slow Training
```bash
# Verificar:
# - Usar SSD ao invés de HDD
# - Aumentar workers (cuidado com RAM)
python scripts/train.py --workers 8

# - Verificar se está usando GPU
python -c "import torch; print(torch.cuda.is_available())"
```

#### 3. Import Errors
```bash
# Verificar PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Ou usar scripts da pasta scripts/
python scripts/train.py  # ao invés de python src/training.py
```

#### 4. Poor mAP Results
```bash
# Possíveis soluções:
# - Aumentar épocas
python scripts/train.py --epochs 500

# - Ajustar learning rate
# Editar configs/hyp.scratch-high.yaml: lr0: 0.005

# - Verificar qualidade dos dados
python -c "
from src.data_preparation import DataPreparator
prep = DataPreparator()
prep.validate_dataset_structure()
"
```

### Comandos de Diagnóstico

```bash
# Verificar instalação completa
python -c "
from src.config import config
from src.utils import check_yolov9_installation
import torch

print('=== DIAGNÓSTICO DO SISTEMA ===')
print(f'Python: {torch.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA disponível: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')

print(f'YOLOv9 instalado: {check_yolov9_installation()}')

paths = config.validate_paths()
print('\\nCaminhos:')
for name, exists in paths.items():
    status = '✓' if exists else '✗'
    print(f'  {status} {name}')
"

# Testar pipeline completo
python -c "
from src.data_preparation import DataPreparator
from src.training import YOLOv9Trainer
from src.validation import YOLOv9Validator
from src.detection import YOLOv9Detector

print('=== TESTE DE PIPELINE ===')
try:
    prep = DataPreparator()
    print('✓ DataPreparator OK')
    
    trainer = YOLOv9Trainer()
    print('✓ YOLOv9Trainer OK')
    
    validator = YOLOv9Validator()
    print('✓ YOLOv9Validator OK')
    
    detector = YOLOv9Detector()
    print('✓ YOLOv9Detector OK')
    
    print('\\n✓ Todos os módulos funcionando!')
    
except Exception as e:
    print(f'✗ Erro: {e}')
"
```

## Configuração para Produção

### Otimização do Modelo

```bash
# Após treinamento, otimizar modelo para inferência
python -c "
import torch
from pathlib import Path

# Carregar modelo
model_path = 'pesos/best.pt'
model = torch.load(model_path, map_location='cpu')

# Salvar apenas os pesos (menor tamanho)
torch.save(model['model'].state_dict(), 'pesos/best_weights_only.pt')

print('Modelo otimizado salvo em: pesos/best_weights_only.pt')
"
```

### Docker (Opcional)

```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# Instalar Python e dependências
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar projeto
WORKDIR /app
COPY . .

# Instalar dependências Python
RUN pip3 install -r requirements.txt

# Comando padrão
CMD ["python3", "scripts/detect.py", "--help"]
```

```bash
# Construir imagem
docker build -t detector-placas .

# Executar detecção
docker run --gpus all -v $(pwd)/imagens:/app/imagens \
    detector-placas python3 scripts/detect.py \
    --weights pesos/best.pt \
    --source imagens/
```

## Próximos Passos

Após a configuração inicial:

1. **Treinar modelo inicial** com dataset pequeno para testar pipeline
2. **Validar resultados** e ajustar hiperparâmetros se necessário
3. **Treinar modelo final** com dataset completo
4. **Implementar pipeline de produção** se necessário
5. **Monitorar performance** e retreinar periodicamente

Para mais detalhes, consulte:
- [Documentação Técnica](TECHNICAL_DOCUMENTATION.md)
- [Referência da API](API_REFERENCE.md)