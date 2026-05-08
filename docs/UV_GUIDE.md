# Guia Completo: Usando `uv` no Projeto

## 📚 Índice
1. [O que é `uv`?](#o-que-é-uv)
2. [Instalação](#instalação)
3. [Primeiros Passos](#primeiros-passos)
4. [Comandos Principais](#comandos-principais)
5. [Gerenciamento de Dependências](#gerenciamento-de-dependências)
6. [Grupos de Dependências](#grupos-de-dependências)
7. [Arquivo `uv.lock`](#arquivo-uvlock)
8. [Arquivo `pyproject.toml`](#arquivo-pyprojecttoml)
9. [Troubleshooting](#troubleshooting)

---

## 🤔 O que é `uv`?

`uv` é um **gerenciador de pacotes Python** moderno, escrito em Rust, que é **extremamente rápido** e oferece uma experiência melhor que `pip`.

### Características Principais

| Recurso | Descrição |
|---------|-----------|
| ⚡ **Velocidade** | 10-100x mais rápido que pip |
| 🔒 **Determinístico** | Reproduz exatamente o mesmo ambiente |
| 📦 **Lock file** | `uv.lock` garante versões exatas |
| 🐍 **Gerencia Python** | Instala versões do Python automaticamente |
| 🔧 **Grupos de deps** | Organize dependências por categoria |
| 📋 **pyproject.toml** | Configuração centralizada |

### Comparação com `pip`

```
pip:  Lento, não determinístico, sem lock file
uv:   Rápido, determinístico, com lock file
```

---

## 📥 Instalação

### Linux/macOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows (PowerShell)

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Verificar Instalação

```bash
uv --version
# uv 0.1.0 (ou versão mais recente)
```

### Adicionar ao PATH (se necessário)

```bash
# Linux/macOS
export PATH="$HOME/.local/bin:$PATH"

# Adicionar ao ~/.bashrc ou ~/.zshrc para persistir
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

---

## 🚀 Primeiros Passos

### 1. Clonar o Projeto

```bash
git clone <url-do-repositorio>
cd detector-placas-yolov9
```

### 2. Sincronizar Dependências

```bash
# Criar ambiente virtual e instalar dependências
uv sync
```

**O que acontece**:
- ✅ Cria `.venv/` (ambiente virtual)
- ✅ Instala todas as dependências
- ✅ Gera/atualiza `uv.lock`

### 3. Ativar Ambiente Virtual

```bash
# Linux/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 4. Executar Comandos

```bash
# Opção 1: Com ambiente ativado
python scripts/train.py

# Opção 2: Sem ativar (usando uv run)
uv run python scripts/train.py
```

---

## 🔧 Comandos Principais

### `uv sync` - Sincronizar Dependências

```bash
# Instalar/atualizar todas as dependências
uv sync

# Sincronizar com grupo específico
uv sync --group dev
uv sync --group torch-cuda121

# Sincronizar com múltiplos grupos
uv sync --group dev --group torch-cuda121

# Atualizar todas as dependências
uv sync --upgrade

# Usar versão específica do Python
uv sync --python 3.11
```

### `uv add` - Adicionar Dependência

```bash
# Adicionar ao grupo principal
uv add numpy

# Adicionar ao grupo de desenvolvimento
uv add --group dev pytest

# Adicionar com versão específica
uv add "torch>=2.0.0"

# Adicionar múltiplas
uv add numpy pandas scipy
```

### `uv remove` - Remover Dependência

```bash
# Remover do grupo principal
uv remove numpy

# Remover do grupo específico
uv remove --group dev pytest
```

### `uv run` - Executar Comando

```bash
# Executar script Python
uv run python scripts/train.py

# Executar com argumentos
uv run python scripts/detect.py --weights pesos/best.pt --source imagens/

# Executar comando Python
uv run python -c "import torch; print(torch.__version__)"

# Executar script diretamente
uv run scripts/train.py
```

### `uv pip` - Compatibilidade com pip

```bash
# Usar comandos pip (compatibilidade)
uv pip list
uv pip show numpy
```

### `uv cache` - Gerenciar Cache

```bash
# Limpar cache
uv cache clean

# Ver informações do cache
uv cache dir
```

---

## 📦 Gerenciamento de Dependências

### Adicionar Dependência Simples

```bash
# Adicionar versão mais recente
uv add requests

# Adicionar versão específica
uv add "requests==2.28.0"

# Adicionar com intervalo de versão
uv add "requests>=2.28.0,<3.0.0"
```

### Atualizar Dependência

```bash
# Atualizar pacote específico
uv add --upgrade numpy

# Atualizar todas
uv sync --upgrade
```

### Remover Dependência

```bash
uv remove numpy
```

### Listar Dependências

```bash
# Listar instaladas
uv pip list

# Mostrar informações de um pacote
uv pip show numpy
```

### Verificar Compatibilidade

```bash
# Verificar se há conflitos
uv sync --check
```

---

## 🔀 Grupos de Dependências

### O que são Grupos?

Grupos permitem organizar dependências por categoria:
- `main`: Dependências principais (sempre instaladas)
- `dev`: Dependências de desenvolvimento (opcional)
- `torch-cuda121`: PyTorch com CUDA 12.1 (opcional)
- `torch-cuda118`: PyTorch com CUDA 11.8 (opcional)
- `torch-cpu`: PyTorch para CPU (opcional)

### Definir Grupos no `pyproject.toml`

```toml
[dependency-groups]
dev = [
    "black>=22.0.0",
    "pytest>=7.0.0",
    "mypy>=0.991",
]
torch-cuda121 = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
]
```

### Usar Grupos

```bash
# Instalar apenas dependências principais
uv sync

# Instalar com grupo dev
uv sync --group dev

# Instalar com PyTorch CUDA 12.1
uv sync --group torch-cuda121

# Instalar múltiplos grupos
uv sync --group dev --group torch-cuda121

# Instalar todos os grupos
uv sync --all-groups
```

### Adicionar a Grupo Específico

```bash
# Adicionar ao grupo dev
uv add --group dev pytest

# Adicionar ao grupo torch-cuda121
uv add --group torch-cuda121 torch
```

---

## 🔒 Arquivo `uv.lock`

### O que é?

`uv.lock` é um arquivo que contém:
- ✅ Versões exatas de todas as dependências
- ✅ Hashes para verificação de integridade
- ✅ Informações de compatibilidade
- ✅ Histórico de mudanças

### Exemplo de Conteúdo

```toml
version = 1
requires-python = ">=3.8"

[[package]]
name = "numpy"
version = "2.4.4"
source = { registry = "https://pypi.org/simple" }
sdist = { url = "...", hash = "sha256:..." }
wheels = [
    { url = "...", hash = "sha256:..." },
]
```

### Benefícios

| Benefício | Descrição |
|-----------|-----------|
| 🔒 **Reprodutibilidade** | Mesmo ambiente em qualquer máquina |
| 🚀 **Velocidade** | Não precisa resolver dependências |
| 📝 **Rastreabilidade** | Histórico via git |
| 🛡️ **Segurança** | Verifica integridade dos pacotes |

### Usar `uv.lock`

```bash
# Sincronizar com versões do lock
uv sync

# Atualizar lock (sem mudar versões)
uv lock

# Forçar atualizar lock
uv lock --upgrade
```

### Fazer Commit do `uv.lock`

```bash
# Adicionar ao git
git add uv.lock

# Fazer commit
git commit -m "Atualizar dependências"

# Push
git push
```

---

## 📋 Arquivo `pyproject.toml`

### Estrutura Básica

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "detector-placas-yolov9"
version = "2.0.0"
description = "Detector de caracteres em placas veiculares"
requires-python = ">=3.8"
dependencies = [
    "numpy>=1.21.0",
    "torch>=2.0.0",
    # ...
]

[dependency-groups]
dev = [
    "pytest>=7.0.0",
    "black>=22.0.0",
]

[tool.black]
line-length = 88

[tool.pytest.ini_options]
testpaths = ["tests"]
```

### Seções Principais

#### `[project]`
Define metadados do projeto:
```toml
[project]
name = "detector-placas-yolov9"
version = "2.0.0"
description = "..."
requires-python = ">=3.8"
dependencies = [...]
```

#### `[dependency-groups]`
Define grupos de dependências:
```toml
[dependency-groups]
dev = [...]
torch-cuda121 = [...]
```

#### `[tool.*]`
Configurações de ferramentas:
```toml
[tool.black]
line-length = 88

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.mypy]
python_version = "3.8"
```

### Editar `pyproject.toml`

```bash
# Editar manualmente
nano pyproject.toml

# Ou usar uv add/remove
uv add novo-pacote
uv remove pacote-antigo
```

---

## 🐛 Troubleshooting

### Erro: "uv: command not found"

**Solução**:
```bash
# Instalar uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Adicionar ao PATH
export PATH="$HOME/.local/bin:$PATH"

# Verificar
uv --version
```

### Erro: "Python version not found"

**Solução**:
```bash
# Instalar versão específica do Python
uv python install 3.11

# Usar versão específica
uv sync --python 3.11
```

### Erro: "Conflicting dependencies"

**Solução**:
```bash
# Limpar cache
uv cache clean

# Sincronizar novamente
uv sync

# Ou forçar atualização
uv sync --upgrade
```

### Ambiente Virtual Corrompido

**Solução**:
```bash
# Remover ambiente virtual
rm -rf .venv

# Sincronizar novamente
uv sync
```

### Atualizar `uv`

```bash
# Atualizar para versão mais recente
uv self update

# Verificar versão
uv --version
```

---

## 📊 Workflow Recomendado

### Desenvolvimento Local

```bash
# 1. Clonar projeto
git clone <url>
cd detector-placas-yolov9

# 2. Sincronizar com grupos dev
uv sync --group dev

# 3. Ativar ambiente
source .venv/bin/activate

# 4. Fazer alterações
# ... editar código ...

# 5. Adicionar nova dependência
uv add novo-pacote

# 6. Fazer commit
git add pyproject.toml uv.lock
git commit -m "Adicionar novo-pacote"

# 7. Push
git push
```

### Produção

```bash
# 1. Clonar projeto
git clone <url>
cd detector-placas-yolov9

# 2. Sincronizar (sem dev)
uv sync

# 3. Executar aplicação
python scripts/train.py
```

### CI/CD

```bash
# GitHub Actions
- name: Setup Python
  uses: astral-sh/setup-uv@v1

- name: Sync dependencies
  run: uv sync --all-groups

- name: Run tests
  run: uv run pytest
```

---

## 🎯 Dicas e Boas Práticas

### 1. Sempre Fazer Commit de `uv.lock`

```bash
git add uv.lock
git commit -m "Atualizar dependências"
```

### 2. Usar Grupos para Organizar

```toml
[dependency-groups]
dev = ["pytest", "black"]
torch-cuda121 = ["torch", "torchvision"]
```

### 3. Especificar Versões Mínimas

```toml
dependencies = [
    "numpy>=1.21.0",  # Versão mínima
    "torch>=2.0.0",
]
```

### 4. Usar `uv run` para Scripts

```bash
# Ao invés de ativar ambiente
uv run python scripts/train.py

# Funciona em qualquer lugar
```

### 5. Limpar Cache Periodicamente

```bash
uv cache clean
```

### 6. Atualizar Dependências Regularmente

```bash
# Verificar atualizações
uv sync --upgrade

# Fazer commit
git add uv.lock
git commit -m "Atualizar dependências"
```

---

## 📚 Recursos Adicionais

- [Documentação Oficial do `uv`](https://docs.astral.sh/uv/)
- [GitHub do `uv`](https://github.com/astral-sh/uv)
- [Comparação: pip vs uv](https://docs.astral.sh/uv/pip/)

---

## ✅ Checklist de Setup

- [ ] Instalar `uv`
- [ ] Clonar repositório
- [ ] Executar `uv sync`
- [ ] Ativar ambiente virtual
- [ ] Verificar instalação: `python -c "import torch"`
- [ ] Fazer commit de `uv.lock`
- [ ] Começar a desenvolver!

Agora você está pronto para usar `uv` no projeto! 🚀