# Detector de Caracteres em Placas Veiculares - YOLOv9 (Refatorado)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![YOLOv9](https://img.shields.io/badge/YOLOv9-Latest-green.svg)](https://github.com/WongKinYiu/yolov9)
[![Code Style](https://img.shields.io/badge/Code%20Style-PEP%208-black.svg)](https://pep8.org)

Projeto refatorado de detecção de caracteres individuais em placas veiculares brasileiras usando YOLOv9, seguindo as melhores práticas da PEP 8 e princípios de código limpo.

## 🚀 Principais Melhorias

### ✨ Refatoração Completa
- **Código modular** organizado em pacotes Python
- **Conformidade PEP 8** com type hints e docstrings
- **Tratamento robusto de erros** e logging estruturado
- **Configurações centralizadas** em classe dedicada
- **API consistente** entre todos os módulos

### 📁 Nova Estrutura
```
projeto/
├── src/                     # 🎯 Código fonte modular
│   ├── config.py           # ⚙️ Configurações centralizadas
│   ├── utils.py            # 🛠️ Utilitários reutilizáveis
│   ├── data_preparation.py # 📊 Preparação de dados
│   ├── training.py         # 🏋️ Treinamento do modelo
│   ├── validation.py       # ✅ Validação e métricas
│   └── detection.py        # 🔍 Detecção e inferência
├── scripts/                # 📜 Scripts de linha de comando
├── docs/                   # 📚 Documentação técnica
└── configs/                # ⚙️ Arquivos de configuração
```

### 🎯 Funcionalidades Avançadas
- **Processamento inteligente** de resultados de detecção
- **Reconstrução automática** do texto da placa
- **Validação de estrutura** do dataset
- **Logging configurável** com múltiplos níveis
- **Tratamento de exceções** específicas por contexto

## 📋 Características do Projeto

### 🎯 Detecção de Classes
- **35 classes**: Dígitos (0-9) + Letras (A-Z, exceto O)
- **Formato brasileiro**: Compatível com placas nacionais
- **Alta precisão**: Otimizado para caracteres individuais

### 🏗️ Arquitetura
- **YOLOv9**: Estado da arte em detecção de objetos
- **Modular**: Componentes independentes e reutilizáveis
- **Escalável**: Fácil extensão e manutenção
- **Documentado**: Documentação técnica completa

## 🚀 Início Rápido

### 1. Instalação com `uv` (Recomendado)

```bash
# Clonar repositório
git clone <url-do-repositorio>
cd detector-placas-yolov9

# Instalar uv (se não tiver)
# Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell):
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Sincronizar dependências e criar ambiente virtual
uv sync

# Ativar ambiente virtual
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Baixar YOLOv9
mkdir -p uvv && cd uvv
git clone https://github.com/WongKinYiu/yolov9.git yolov9-main
cd ..
```

### 2. Instalação Tradicional (pip)

```bash
# Clonar repositório
git clone <url-do-repositorio>
cd detector-placas-yolov9

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Instalar PyTorch (GPU CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Instalar dependências
pip install -r requirements_refatorado.txt

# Baixar YOLOv9
mkdir -p uvv && cd uvv
git clone https://github.com/WongKinYiu/yolov9.git yolov9-main
cd ..
```

### 2. Preparar Dados
```bash
# Opção 1: A partir de ZIPs
python scripts/prepare_data.py --zips-dir "caminho/para/zips"

# Opção 2: Validar estrutura existente
python -c "from src.data_preparation import DataPreparator; DataPreparator().validate_dataset_structure()"
```

### 3. Treinar Modelo
```bash
# Treinamento básico
python scripts/train.py

# Treinamento customizado
python scripts/train.py --epochs 300 --batch-size 16 --device 0
```

### 4. Validar Resultados
```bash
python scripts/validate.py --weights pesos/best.pt --verbose
```

### 5. Executar Detecção
```bash
# Detecção em pasta
python scripts/detect.py --weights pesos/best.pt --source imagens/

# Detecção com processamento
python scripts/detect.py --weights pesos/best.pt --source imagens/ --save-txt --save-conf
```

## 🛠️ Uso Avançado

### Configuração Personalizada
```python
from src.config import config
from src.training import YOLOv9Trainer

# Modificar configurações
config.DEFAULT_EPOCHS = 500
config.DEFAULT_BATCH_SIZE = 32

# Treinar com configurações customizadas
trainer = YOLOv9Trainer()
trainer.train(
    epochs=500,
    batch_size=32,
    device="0",
    name="detector_avancado"
)
```

### Processamento de Resultados
```python
from src.detection import PlateCharacterProcessor
from pathlib import Path

# Processar resultados de detecção
processor = PlateCharacterProcessor()
results = processor.process_detection_results(
    results_dir=Path("runs/detect/exp/"),
    confidence_threshold=0.6
)

# Exibir placas detectadas
for image_name, data in results.items():
    print(f"{image_name}: {data['plate_text']} (confiança: {data['confidence']:.2f})")
```

### Pipeline Completo
```python
from src.data_preparation import prepare_data_from_zips
from src.training import YOLOv9Trainer
from src.validation import YOLOv9Validator
from src.detection import YOLOv9Detector

# Pipeline automatizado
prepare_data_from_zips("zips/")

trainer = YOLOv9Trainer()
trainer.train(epochs=200)

validator = YOLOv9Validator()
validator.validate(weights="runs/train/detector_placas/weights/best.pt")

detector = YOLOv9Detector()
detector.detect(weights="runs/train/detector_placas/weights/best.pt", source="test_images/")
```

## 📊 Configurações

### Hiperparâmetros (configs/hyp.scratch-high.yaml)
```yaml
lr0: 0.01                    # Taxa de aprendizado inicial
momentum: 0.937              # Momentum do SGD
weight_decay: 0.0005         # Decaimento de peso
box: 7.5                     # Peso da loss de bounding box
cls: 0.5                     # Peso da loss de classificação
mosaic: 1.0                  # Probabilidade de mosaic augmentation
mixup: 0.15                  # Probabilidade de mixup
```

### Dataset (configs/dados-placas.yaml)
```yaml
path: ../../dados            # Caminho raiz
train: treino               # Dados de treinamento
val: validacao              # Dados de validação
test: teste                 # Dados de teste
nc: 35                      # Número de classes
names: ['0', '1', ..., 'Z'] # Nomes das classes
```

## 🔧 Troubleshooting

### Problemas Comuns

#### CUDA Out of Memory
```bash
# Reduzir batch size
python scripts/train.py --batch-size 4

# Reduzir tamanho da imagem
python scripts/train.py --img-size 416
```

#### Treinamento Lento
```bash
# Verificar GPU
python -c "import torch; print(torch.cuda.is_available())"

# Aumentar workers (cuidado com RAM)
python scripts/train.py --workers 8
```

#### Baixa Precisão
```bash
# Aumentar épocas
python scripts/train.py --epochs 500

# Verificar qualidade dos dados
python -c "from src.data_preparation import DataPreparator; DataPreparator().validate_dataset_structure()"
```

### Diagnóstico do Sistema
```bash
python -c "
from src.config import config
from src.utils import check_yolov9_installation
import torch

print('=== DIAGNÓSTICO ===')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'YOLOv9: {check_yolov9_installation()}')
print('Caminhos:', config.validate_paths())
"
```

## 📦 Gerenciamento de Dependências com `uv`

### O que é `uv`?

`uv` é um gerenciador de pacotes Python **extremamente rápido**, escrito em Rust. É uma alternativa moderna ao `pip` e `poetry`.

**Vantagens**:
- ⚡ **10-100x mais rápido** que pip
- 🔒 **Determinístico**: `uv.lock` garante reprodutibilidade
- 📦 **Gerenciamento de dependências**: Resolve conflitos automaticamente
- 🐍 **Gerenciamento de Python**: Instala versões do Python automaticamente

### Usando `uv` no Projeto

#### Sincronizar Dependências
```bash
# Instalar/atualizar todas as dependências
uv sync

# Instalar com grupo específico (dev, torch-cuda121, etc)
uv sync --group dev
uv sync --group torch-cuda121
```

#### Adicionar Novas Dependências
```bash
# Adicionar ao grupo principal
uv add numpy

# Adicionar ao grupo de desenvolvimento
uv add --group dev pytest

# Adicionar com versão específica
uv add "torch>=2.0.0"
```

#### Remover Dependências
```bash
uv remove numpy
uv remove --group dev pytest
```

#### Atualizar Dependências
```bash
# Atualizar todas
uv sync --upgrade

# Atualizar pacote específico
uv add --upgrade numpy
```

#### Executar Comandos no Ambiente Virtual
```bash
# Executar script
uv run python scripts/train.py

# Executar comando
uv run python -c "import torch; print(torch.__version__)"

# Executar com argumentos
uv run python scripts/detect.py --weights pesos/best.pt --source imagens/
```

### Arquivo `uv.lock`

O arquivo `uv.lock` é gerado automaticamente e contém:
- ✅ Versões exatas de todas as dependências
- ✅ Hashes para verificação de integridade
- ✅ Informações de compatibilidade

**Benefícios**:
- 🔒 **Reprodutibilidade**: Mesmo ambiente em qualquer máquina
- 🚀 **Velocidade**: Não precisa resolver dependências novamente
- 📝 **Rastreabilidade**: Histórico de mudanças via git

### Arquivo `pyproject.toml`

O arquivo `pyproject.toml` define:
- 📦 Metadados do projeto
- 📋 Dependências principais
- 🔧 Grupos de dependências (dev, torch-cuda121, etc)
- ⚙️ Configurações de ferramentas (black, mypy, pytest, etc)

**Estrutura**:
```toml
[project]
name = "detector-placas-yolov9"
version = "2.0.0"
requires-python = ">=3.8"
dependencies = [...]

[dependency-groups]
dev = [...]
torch-cuda121 = [...]

[tool.black]
line-length = 88

[tool.pytest.ini_options]
testpaths = ["tests"]
```

### Comparação: `pip` vs `uv`

| Aspecto | pip | uv |
|---------|-----|-----|
| **Velocidade** | Lenta | 10-100x mais rápida |
| **Lock file** | Não | Sim (uv.lock) |
| **Determinístico** | Não | Sim |
| **Gerenciar Python** | Não | Sim |
| **Grupos de deps** | Não | Sim |
| **Compatibilidade** | Boa | Excelente |

### Instalação do `uv`

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Verificar instalação
uv --version
```

### Workflow Recomendado com `uv`

```bash
# 1. Clonar projeto
git clone <url>
cd detector-placas-yolov9

# 2. Sincronizar dependências
uv sync

# 3. Ativar ambiente virtual
source .venv/bin/activate

# 4. Executar scripts
python scripts/train.py

# 5. Adicionar nova dependência
uv add novo-pacote

# 6. Fazer commit do uv.lock
git add uv.lock
git commit -m "Atualizar dependências"
```

### Troubleshooting com `uv`

#### Erro: "uv: command not found"
```bash
# Instalar uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Adicionar ao PATH (se necessário)
export PATH="$HOME/.local/bin:$PATH"
```

#### Limpar cache
```bash
uv cache clean
```

#### Reinstalar dependências
```bash
rm uv.lock
uv sync
```

#### Usar versão específica do Python
```bash
uv sync --python 3.11
```

## 📚 Documentação

### Documentos Disponíveis
- **[Documentação Técnica](docs/TECHNICAL_DOCUMENTATION.md)**: Arquitetura e implementação detalhada
- **[Referência da API](docs/API_REFERENCE.md)**: Documentação completa das classes e métodos
- **[Guia de Configuração](docs/SETUP_GUIDE.md)**: Instalação e configuração passo a passo
- **[Explicação da Arquitetura](docs/EXPLICACAO_ARQUITETURA.md)**: Entenda a estrutura modular do projeto
- **[Exemplos Práticos](docs/EXEMPLOS_PRATICOS.md)**: Exemplos de uso como script e como módulo

### Exemplos de Código
Consulte a documentação da API para exemplos completos de uso de cada módulo.

### Entender a Diferença: Script vs Módulo
O projeto pode ser usado de **duas formas**:

1. **Como Script (CLI)**: Executar via linha de comando
   ```bash
   python scripts/validate.py --weights pesos/best.pt
   ```

2. **Como Módulo (Programático)**: Importar em seus próprios scripts
   ```python
   from src.validation import YOLOv9Validator
   validator = YOLOv9Validator()
   validator.validate(weights="pesos/best.pt")
   ```

Veja [Exemplos Práticos](docs/EXEMPLOS_PRATICOS.md) para entender melhor!

## 🎯 Melhorias Implementadas

### Código Limpo
- ✅ **PEP 8 compliance** completa
- ✅ **Type hints** em todas as funções
- ✅ **Docstrings** no formato Google/NumPy
- ✅ **Tratamento de exceções** específico
- ✅ **Logging estruturado** com níveis apropriados

### Arquitetura
- ✅ **Separação de responsabilidades** clara
- ✅ **Configurações centralizadas** em classe dedicada
- ✅ **Módulos independentes** e reutilizáveis
- ✅ **Interface consistente** entre componentes
- ✅ **Extensibilidade** para futuras funcionalidades

### Funcionalidades
- ✅ **Validação automática** de estrutura de dados
- ✅ **Processamento inteligente** de resultados
- ✅ **Reconstrução de placas** a partir de caracteres
- ✅ **Métricas de confiança** por detecção
- ✅ **Scripts de linha de comando** organizados

## 🤝 Contribuição

### Padrões de Desenvolvimento
1. **Seguir PEP 8** rigorosamente
2. **Adicionar type hints** em todas as funções
3. **Escrever docstrings** descritivas
4. **Implementar testes** para novas funcionalidades
5. **Usar logging** ao invés de print

### Estrutura de Commits
```
tipo(escopo): descrição breve

Descrição detalhada das alterações.

- Alteração 1
- Alteração 2
```

Tipos: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## 📄 Licença

Este projeto utiliza o YOLOv9 sob sua licença original. Consulte o repositório oficial para detalhes.

## 🙏 Créditos

- **YOLOv9**: [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)
- **Dataset**: Placas veiculares brasileiras
- **Refatoração**: Implementação seguindo melhores práticas Python

---

**Versão Refatorada**: Código mais limpo, documentado e escalável para detecção de placas veiculares! 🚗✨