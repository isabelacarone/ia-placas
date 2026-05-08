# Makefile para Detector de Placas YOLOv9
# Facilita execução de tarefas comuns de desenvolvimento

.PHONY: help install install-dev setup clean lint format test train validate detect docs

# Configurações
PYTHON := python3
PIP := pip3
VENV := venv
SRC_DIR := src
SCRIPTS_DIR := scripts
DOCS_DIR := docs

# Cores para output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

help: ## Mostra esta mensagem de ajuda
	@echo "$(BLUE)Detector de Placas YOLOv9 - Comandos Disponíveis$(NC)"
	@echo "=================================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# ============================================================================
# SETUP E INSTALAÇÃO
# ============================================================================

setup: ## Configuração inicial completa do projeto
	@echo "$(BLUE)Configurando projeto...$(NC)"
	$(MAKE) create-venv
	$(MAKE) install-pytorch
	$(MAKE) install
	$(MAKE) download-yolov9
	$(MAKE) verify-setup
	@echo "$(GREEN)✓ Configuração concluída!$(NC)"

create-venv: ## Cria ambiente virtual
	@echo "$(YELLOW)Criando ambiente virtual...$(NC)"
	$(PYTHON) -m venv $(VENV)
	@echo "$(GREEN)✓ Ambiente virtual criado em $(VENV)/$(NC)"
	@echo "$(YELLOW)Ative com: source $(VENV)/bin/activate$(NC)"

install-pytorch: ## Instala PyTorch (GPU CUDA 12.1)
	@echo "$(YELLOW)Instalando PyTorch...$(NC)"
	$(PIP) install torch torchvision --index-url https://download.pytorch.org/whl/cu121

install-pytorch-cpu: ## Instala PyTorch (CPU apenas)
	@echo "$(YELLOW)Instalando PyTorch (CPU)...$(NC)"
	$(PIP) install torch torchvision --index-url https://download.pytorch.org/whl/cpu

install: ## Instala dependências do projeto
	@echo "$(YELLOW)Instalando dependências...$(NC)"
	$(PIP) install -r requirements_refatorado.txt

install-dev: ## Instala dependências de desenvolvimento
	@echo "$(YELLOW)Instalando dependências de desenvolvimento...$(NC)"
	$(PIP) install -r requirements_refatorado.txt
	$(PIP) install black flake8 mypy isort pytest pytest-cov

download-yolov9: ## Baixa código fonte do YOLOv9
	@echo "$(YELLOW)Baixando YOLOv9...$(NC)"
	mkdir -p uvv
	cd uvv && git clone https://github.com/WongKinYiu/yolov9.git yolov9-main
	@echo "$(GREEN)✓ YOLOv9 baixado em uvv/yolov9-main/$(NC)"

verify-setup: ## Verifica se a instalação está correta
	@echo "$(YELLOW)Verificando instalação...$(NC)"
	$(PYTHON) -c "from src.utils import check_yolov9_installation; print('✓ YOLOv9 OK' if check_yolov9_installation() else '✗ YOLOv9 Erro')"
	$(PYTHON) -c "import torch; print('✓ CUDA OK' if torch.cuda.is_available() else '✗ CUDA Indisponível')"
	$(PYTHON) -c "from src.config import config; print('✓ Configuração OK')"

# ============================================================================
# DESENVOLVIMENTO
# ============================================================================

lint: ## Executa linting do código
	@echo "$(YELLOW)Executando linting...$(NC)"
	flake8 $(SRC_DIR) $(SCRIPTS_DIR) --max-line-length=88 --extend-ignore=E203,W503
	mypy $(SRC_DIR) --ignore-missing-imports

format: ## Formata código com black e isort
	@echo "$(YELLOW)Formatando código...$(NC)"
	black $(SRC_DIR) $(SCRIPTS_DIR)
	isort $(SRC_DIR) $(SCRIPTS_DIR)

test: ## Executa testes
	@echo "$(YELLOW)Executando testes...$(NC)"
	pytest tests/ -v --cov=$(SRC_DIR) --cov-report=html --cov-report=term

clean: ## Remove arquivos temporários e cache
	@echo "$(YELLOW)Limpando arquivos temporários...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/

# ============================================================================
# PIPELINE DE DADOS E TREINAMENTO
# ============================================================================

prepare-data: ## Prepara dados a partir de ZIPs (requer ZIPS_DIR)
	@if [ -z "$(ZIPS_DIR)" ]; then \
		echo "$(RED)Erro: Especifique ZIPS_DIR. Exemplo: make prepare-data ZIPS_DIR=/caminho/para/zips$(NC)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Preparando dados de $(ZIPS_DIR)...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/prepare_data.py --zips-dir "$(ZIPS_DIR)"

validate-data: ## Valida estrutura do dataset
	@echo "$(YELLOW)Validando estrutura do dataset...$(NC)"
	$(PYTHON) -c "from src.data_preparation import DataPreparator; DataPreparator().validate_dataset_structure()"

train: ## Treina modelo (configurações padrão)
	@echo "$(YELLOW)Iniciando treinamento...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/train.py

train-quick: ## Treinamento rápido (poucas épocas para teste)
	@echo "$(YELLOW)Treinamento rápido (10 épocas)...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/train.py --epochs 10 --batch-size 4

train-gpu: ## Treinamento otimizado para GPU
	@echo "$(YELLOW)Treinamento otimizado para GPU...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/train.py --epochs 300 --batch-size 16 --device 0

train-cpu: ## Treinamento em CPU (não recomendado)
	@echo "$(YELLOW)Treinamento em CPU...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/train.py --epochs 50 --batch-size 2 --device cpu

validate: ## Valida modelo treinado (requer WEIGHTS)
	@if [ -z "$(WEIGHTS)" ]; then \
		echo "$(RED)Erro: Especifique WEIGHTS. Exemplo: make validate WEIGHTS=pesos/best.pt$(NC)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Validando modelo $(WEIGHTS)...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/validate.py --weights "$(WEIGHTS)" --verbose

detect: ## Executa detecção (requer WEIGHTS e SOURCE)
	@if [ -z "$(WEIGHTS)" ] || [ -z "$(SOURCE)" ]; then \
		echo "$(RED)Erro: Especifique WEIGHTS e SOURCE. Exemplo: make detect WEIGHTS=pesos/best.pt SOURCE=imagens/$(NC)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Executando detecção em $(SOURCE)...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/detect.py --weights "$(WEIGHTS)" --source "$(SOURCE)" --save-txt

# ============================================================================
# PIPELINE COMPLETO
# ============================================================================

pipeline-full: ## Pipeline completo (requer ZIPS_DIR)
	@if [ -z "$(ZIPS_DIR)" ]; then \
		echo "$(RED)Erro: Especifique ZIPS_DIR. Exemplo: make pipeline-full ZIPS_DIR=/caminho/para/zips$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Executando pipeline completo...$(NC)"
	$(MAKE) prepare-data ZIPS_DIR="$(ZIPS_DIR)"
	$(MAKE) train-gpu
	$(MAKE) validate WEIGHTS=uvv/yolov9-main/runs/train/detector_placas/weights/best.pt
	@echo "$(GREEN)✓ Pipeline completo concluído!$(NC)"

pipeline-test: ## Pipeline de teste rápido
	@echo "$(BLUE)Executando pipeline de teste...$(NC)"
	$(MAKE) validate-data
	$(MAKE) train-quick
	@echo "$(GREEN)✓ Pipeline de teste concluído!$(NC)"

# ============================================================================
# DOCUMENTAÇÃO
# ============================================================================

docs: ## Gera documentação
	@echo "$(YELLOW)Gerando documentação...$(NC)"
	@echo "Documentação disponível em:"
	@echo "  - $(DOCS_DIR)/TECHNICAL_DOCUMENTATION.md"
	@echo "  - $(DOCS_DIR)/API_REFERENCE.md"
	@echo "  - $(DOCS_DIR)/SETUP_GUIDE.md"

docs-serve: ## Serve documentação localmente (se usando mkdocs)
	@echo "$(YELLOW)Servindo documentação...$(NC)"
	@if command -v mkdocs >/dev/null 2>&1; then \
		mkdocs serve; \
	else \
		echo "$(RED)mkdocs não instalado. Instale com: pip install mkdocs$(NC)"; \
	fi

# ============================================================================
# UTILITÁRIOS
# ============================================================================

status: ## Mostra status do projeto
	@echo "$(BLUE)Status do Projeto$(NC)"
	@echo "=================="
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Pip: $(shell $(PIP) --version)"
	@echo "Diretório atual: $(shell pwd)"
	@echo ""
	@echo "Estrutura:"
	@ls -la | grep -E "(src|scripts|docs|configs|dados|pesos|uvv)"
	@echo ""
	@echo "Verificações:"
	@$(PYTHON) -c "from src.utils import check_yolov9_installation; print('✓ YOLOv9 OK' if check_yolov9_installation() else '✗ YOLOv9 Erro')" 2>/dev/null || echo "✗ Módulos não instalados"
	@$(PYTHON) -c "import torch; print('✓ CUDA OK' if torch.cuda.is_available() else '✗ CUDA Indisponível')" 2>/dev/null || echo "✗ PyTorch não instalado"

tensorboard: ## Inicia TensorBoard para visualizar treinamento
	@echo "$(YELLOW)Iniciando TensorBoard...$(NC)"
	tensorboard --logdir uvv/yolov9-main/runs/train --port 6006
	@echo "$(GREEN)TensorBoard disponível em: http://localhost:6006$(NC)"

monitor-gpu: ## Monitora uso da GPU
	@echo "$(YELLOW)Monitorando GPU...$(NC)"
	watch -n 1 nvidia-smi

# ============================================================================
# EXEMPLOS DE USO
# ============================================================================

example-setup: ## Exemplo completo de configuração
	@echo "$(BLUE)Exemplo de configuração completa:$(NC)"
	@echo "1. make setup"
	@echo "2. source venv/bin/activate"
	@echo "3. make prepare-data ZIPS_DIR=/caminho/para/zips"
	@echo "4. make train-gpu"
	@echo "5. make validate WEIGHTS=uvv/yolov9-main/runs/train/detector_placas/weights/best.pt"
	@echo "6. make detect WEIGHTS=pesos/best.pt SOURCE=imagens/"

example-development: ## Exemplo de fluxo de desenvolvimento
	@echo "$(BLUE)Exemplo de fluxo de desenvolvimento:$(NC)"
	@echo "1. make install-dev"
	@echo "2. make format"
	@echo "3. make lint"
	@echo "4. make test"
	@echo "5. make train-quick"

# ============================================================================
# INFORMAÇÕES
# ============================================================================

info: ## Mostra informações do projeto
	@echo "$(BLUE)Detector de Placas YOLOv9 - Versão Refatorada$(NC)"
	@echo "=============================================="
	@echo "Projeto de detecção de caracteres em placas veiculares"
	@echo "usando YOLOv9 com código refatorado seguindo PEP 8."
	@echo ""
	@echo "Estrutura:"
	@echo "  src/          - Código fonte modular"
	@echo "  scripts/      - Scripts de linha de comando"
	@echo "  docs/         - Documentação técnica"
	@echo "  configs/      - Arquivos de configuração"
	@echo ""
	@echo "Comandos principais:"
	@echo "  make setup    - Configuração inicial"
	@echo "  make train    - Treinar modelo"
	@echo "  make validate - Validar modelo"
	@echo "  make detect   - Executar detecção"
	@echo ""
	@echo "Para ajuda completa: make help"