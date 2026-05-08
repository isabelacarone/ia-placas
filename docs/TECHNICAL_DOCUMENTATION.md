# Documentação Técnica - Detector de Placas YOLOv9

## Visão Geral

Este projeto implementa um sistema de detecção de caracteres individuais em placas veiculares brasileiras usando YOLOv9. O sistema foi refatorado seguindo as melhores práticas da PEP 8 e princípios de código limpo.

## Arquitetura do Projeto

### Estrutura de Diretórios

```
projeto/
├── src/                          # Código fonte principal
│   ├── __init__.py              # Inicialização do pacote
│   ├── config.py                # Configurações centralizadas
│   ├── utils.py                 # Utilitários e funções auxiliares
│   ├── data_preparation.py      # Preparação e organização dos dados
│   ├── training.py              # Treinamento do modelo
│   ├── validation.py            # Validação e métricas
│   └── detection.py             # Detecção e inferência
├── scripts/                     # Scripts de linha de comando
│   ├── prepare_data.py          # Script para preparar dados
│   ├── train.py                 # Script de treinamento
│   ├── validate.py              # Script de validação
│   └── detect.py                # Script de detecção
├── configs/                     # Arquivos de configuração
│   ├── dados-placas.yaml        # Configuração do dataset
│   └── hyp.scratch-high.yaml    # Hiperparâmetros
├── dados/                       # Dataset organizado
│   ├── treino/                  # Dados de treinamento
│   ├── validacao/               # Dados de validação
│   └── teste/                   # Dados de teste
├── pesos/                       # Modelos treinados
├── uvv/yolov9-main/            # Código fonte do YOLOv9
└── docs/                       # Documentação
```

## Módulos Principais

### 1. config.py - Configurações Centralizadas

**Propósito**: Centralizar todas as configurações, constantes e caminhos do projeto.

**Classe Principal**: `ProjectConfig`
- Gerencia caminhos de diretórios e arquivos
- Define configurações padrão para treinamento, validação e detecção
- Configura variáveis de ambiente necessárias
- Valida instalação do YOLOv9

**Configurações Importantes**:
```python
# Classes detectadas (35 total)
NUM_CLASSES = 35
CLASS_NAMES = ['0'-'9', 'A'-'Z' (exceto 'O')]

# Configurações padrão
DEFAULT_EPOCHS = 300
DEFAULT_BATCH_SIZE = 8
DEFAULT_IMG_SIZE = 640
DEFAULT_CONF_THRESHOLD = 0.25
```

### 2. utils.py - Utilitários

**Propósito**: Fornecer funções auxiliares reutilizáveis em todo o projeto.

**Funções Principais**:
- `setup_logging()`: Configura sistema de logging
- `validate_device()`: Valida especificação de dispositivo GPU/CPU
- `print_header()`: Formatação de cabeçalhos
- `check_yolov9_installation()`: Verifica instalação do YOLOv9
- `parse_yolo_label()`: Parseia arquivos de label do YOLO

### 3. data_preparation.py - Preparação de Dados

**Propósito**: Gerenciar preparação, organização e validação dos dados.

**Classe Principal**: `DataPreparator`

**Funcionalidades**:
- Extração de arquivos ZIP do dataset
- Validação da estrutura de diretórios
- Correção automática de nomes de pastas
- Geração de estatísticas do dataset
- Criação de arquivo YAML de configuração

**Exemplo de Uso**:
```python
from src.data_preparation import DataPreparator

preparator = DataPreparator()
preparator.extract_datasets(Path("caminho/para/zips"))
stats = preparator.validate_dataset_structure()
```

### 4. training.py - Treinamento

**Propósito**: Gerenciar o processo de treinamento do modelo YOLOv9.

**Classe Principal**: `YOLOv9Trainer`

**Funcionalidades**:
- Validação de ambiente e configurações
- Construção de comandos de treinamento
- Execução do processo de treinamento
- Logging detalhado do progresso

**Parâmetros Configuráveis**:
- Número de épocas
- Tamanho do batch
- Dispositivo (GPU/CPU)
- Pesos pré-treinados
- Retomada de treinamento

### 5. validation.py - Validação

**Propósito**: Executar validação do modelo e gerar métricas de performance.

**Classe Principal**: `YOLOv9Validator`

**Métricas Geradas**:
- mAP (mean Average Precision)
- Precision por classe
- Recall por classe
- F1-Score
- Curvas de confiança

### 6. detection.py - Detecção

**Propósito**: Executar inferência em imagens e processar resultados.

**Classes Principais**:
- `YOLOv9Detector`: Execução de detecção
- `PlateCharacterProcessor`: Processamento de resultados

**Funcionalidades**:
- Detecção em imagens individuais ou lotes
- Processamento de resultados
- Reconstrução de texto da placa
- Cálculo de confiança média

## Fluxo de Trabalho

### 1. Preparação dos Dados

```bash
# Extrair e organizar dados
python scripts/prepare_data.py --zips-dir "caminho/para/zips"
```

**Processo**:
1. Extração dos arquivos ZIP
2. Organização em estrutura YOLO
3. Validação de correspondência imagem-label
4. Geração de estatísticas

### 2. Treinamento

```bash
# Treinamento básico
python scripts/train.py --epochs 300 --batch-size 8

# Treinamento com GPU específica
python scripts/train.py --device 0 --epochs 500

# Retomar treinamento
python scripts/train.py --resume pesos/last.pt
```

**Processo**:
1. Validação de ambiente
2. Carregamento de configurações
3. Execução do treinamento YOLOv9
4. Salvamento de checkpoints

### 3. Validação

```bash
# Validação básica
python scripts/validate.py --weights pesos/best.pt

# Validação detalhada
python scripts/validate.py --weights pesos/best.pt --verbose
```

**Processo**:
1. Carregamento do modelo
2. Execução em dataset de validação
3. Cálculo de métricas
4. Geração de relatórios

### 4. Detecção

```bash
# Detecção em pasta
python scripts/detect.py --weights pesos/best.pt --source imagens/

# Detecção com configurações customizadas
python scripts/detect.py --weights pesos/best.pt --source imagem.jpg --conf 0.5 --save-txt
```

**Processo**:
1. Carregamento do modelo
2. Processamento das imagens
3. Aplicação de NMS
4. Salvamento de resultados

## Configurações Avançadas

### Hiperparâmetros (hyp.scratch-high.yaml)

```yaml
# Learning rate
lr0: 0.01                    # Taxa de aprendizado inicial
lrf: 0.01                    # Taxa de aprendizado final

# Otimização
momentum: 0.937              # Momentum do SGD
weight_decay: 0.0005         # Decaimento de peso

# Loss weights
box: 7.5                     # Peso da loss de bounding box
cls: 0.5                     # Peso da loss de classificação
obj: 0.7                     # Peso da loss de objetividade

# Data augmentation
hsv_h: 0.015                 # Augmentation de matiz
hsv_s: 0.7                   # Augmentation de saturação
hsv_v: 0.4                   # Augmentation de valor
mosaic: 1.0                  # Probabilidade de mosaic
mixup: 0.15                  # Probabilidade de mixup
```

### Dataset (dados-placas.yaml)

```yaml
# Caminhos
path: ../../dados            # Caminho raiz
train: treino               # Pasta de treinamento
val: validacao              # Pasta de validação
test: teste                 # Pasta de teste

# Classes
nc: 35                      # Número de classes
names: ['0', '1', ..., 'Z'] # Nomes das classes
```

## Tratamento de Erros

### Validações Implementadas

1. **Instalação do YOLOv9**: Verifica se todos os scripts necessários existem
2. **Arquivos de Configuração**: Valida existência de YAMLs
3. **Pesos do Modelo**: Confirma existência antes de usar
4. **Dispositivo**: Normaliza especificação de GPU/CPU
5. **Estrutura de Dados**: Verifica correspondência imagem-label

### Logging

O sistema implementa logging estruturado com diferentes níveis:

```python
# Configuração de logging
logger = setup_logging(
    level=logging.INFO,
    log_file=Path("logs/detector.log"),
    format_string="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Uso
logger.info("Iniciando treinamento...")
logger.warning("Arquivo não encontrado")
logger.error("Erro crítico durante execução")
```

## Extensibilidade

### Adicionando Novas Funcionalidades

1. **Novos Processadores**: Implementar classes que herdam de base classes
2. **Métricas Customizadas**: Adicionar ao módulo de validação
3. **Augmentations**: Modificar hiperparâmetros YAML
4. **Pós-processamento**: Estender `PlateCharacterProcessor`

### Exemplo de Extensão

```python
class CustomPlateProcessor(PlateCharacterProcessor):
    """Processador customizado com validação de formato brasileiro."""
    
    def validate_brazilian_format(self, plate_text: str) -> bool:
        """Valida se placa segue formato brasileiro."""
        # Implementar validação específica
        pass
    
    def process_detection_results(self, results_dir: Path) -> Dict:
        """Processa com validação adicional."""
        results = super().process_detection_results(results_dir)
        
        # Aplicar validações customizadas
        for image_name, data in results.items():
            if not self.validate_brazilian_format(data['plate_text']):
                data['valid_format'] = False
        
        return results
```

## Performance e Otimização

### Recomendações de Hardware

- **GPU**: NVIDIA com CUDA 11.8+ e 8GB+ VRAM
- **RAM**: 16GB+ para datasets grandes
- **Storage**: SSD para melhor I/O durante treinamento

### Otimizações Implementadas

1. **Batch Size Dinâmico**: Ajuste automático baseado na GPU
2. **Workers Otimizados**: Configuração baseada no número de CPUs
3. **Caching**: Cache de imagens para acelerar treinamento
4. **Mixed Precision**: Suporte a FP16 para GPUs compatíveis

### Monitoramento

```python
# Métricas durante treinamento
- Loss (box, obj, cls)
- mAP@0.5 e mAP@0.5:0.95
- Precision e Recall
- Tempo por época
- Uso de memória GPU
```

## Troubleshooting

### Problemas Comuns

1. **CUDA Out of Memory**: Reduzir batch_size
2. **Slow Training**: Verificar workers e SSD
3. **Poor mAP**: Ajustar hiperparâmetros ou aumentar dados
4. **Import Errors**: Verificar PYTHONPATH e instalação

### Comandos de Diagnóstico

```bash
# Verificar instalação
python -c "from src.utils import check_yolov9_installation; print(check_yolov9_installation())"

# Validar configurações
python -c "from src.config import config; print(config.validate_paths())"

# Testar GPU
python -c "import torch; print(torch.cuda.is_available())"
```

## Contribuição

### Padrões de Código

1. **PEP 8**: Seguir rigorosamente
2. **Type Hints**: Usar em todas as funções
3. **Docstrings**: Formato Google/NumPy
4. **Logging**: Usar logger ao invés de print
5. **Testes**: Implementar para novas funcionalidades

### Estrutura de Commits

```
tipo(escopo): descrição breve

Descrição detalhada do que foi alterado e por quê.

- Item 1 alterado
- Item 2 adicionado
- Item 3 removido
```

Tipos: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## Licença e Créditos

- **YOLOv9**: [Repositório Original](https://github.com/WongKinYiu/yolov9)
- **Dataset**: Placas veiculares brasileiras
- **Implementação**: Refatoração seguindo boas práticas Python