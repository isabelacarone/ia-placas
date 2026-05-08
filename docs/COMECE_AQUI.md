# 🎯 COMECE AQUI - Seu Guia de Navegação

## 👋 Bem-vindo!

Você tem um projeto **completamente refatorado** com:
- ✅ Código modular e bem organizado
- ✅ Documentação técnica completa
- ✅ Configuração moderna com `uv`
- ✅ Pronto para produção

## 🚀 Comece em 3 Passos

### 1️⃣ Setup (5 minutos)
```bash
# Instalar uv (se não tiver)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clonar e configurar
git clone <url>
cd detector-placas-yolov9
uv sync
source .venv/bin/activate
```

### 2️⃣ Entender a Estrutura
Leia: **[docs/EXPLICACAO_ARQUITETURA.md](docs/EXPLICACAO_ARQUITETURA.md)**

Aprenda:
- O que é um módulo vs script
- Como o código está organizado
- Como usar como script ou módulo

### 3️⃣ Ver Exemplos
Leia: **[docs/EXEMPLOS_PRATICOS.md](docs/EXEMPLOS_PRATICOS.md)**

Veja:
- Exemplos de uso como CLI
- Exemplos de uso como módulo Python
- Casos de uso reais

---

## 📚 Documentação por Necessidade

### ⚡ Preciso de Setup Rápido
→ **[QUICK_START.md](QUICK_START.md)** (5 min)

### 🤔 Não Entendo a Estrutura
→ **[docs/EXPLICACAO_ARQUITETURA.md](docs/EXPLICACAO_ARQUITETURA.md)** (15 min)

### 💻 Quero Ver Exemplos
→ **[docs/EXEMPLOS_PRATICOS.md](docs/EXEMPLOS_PRATICOS.md)** (20 min)

### 🐍 Quero Usar como Módulo Python
→ **[docs/API_REFERENCE.md](docs/API_REFERENCE.md)** (30 min)

### ⚙️ Quero Entender `uv`
→ **[docs/UV_GUIDE.md](docs/UV_GUIDE.md)** (20 min)

### 🔧 Preciso de Setup Detalhado
→ **[docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md)** (30 min)

### 📖 Quero Documentação Técnica Completa
→ **[docs/TECHNICAL_DOCUMENTATION.md](docs/TECHNICAL_DOCUMENTATION.md)** (45 min)

### 📝 Quero Ver o que Mudou
→ **[CHANGELOG.md](CHANGELOG.md)** (10 min)

### 📊 Quero Entender a Estrutura Final
→ **[ESTRUTURA_FINAL.md](ESTRUTURA_FINAL.md)** (15 min)

---

## 🎯 Fluxo Recomendado

```
1. QUICK_START.md (5 min)
   ↓
2. EXPLICACAO_ARQUITETURA.md (15 min)
   ↓
3. EXEMPLOS_PRATICOS.md (20 min)
   ↓
4. Comece a usar! 🚀
```

---

## 🔥 Comandos Essenciais

### Preparar Dados
```bash
python scripts/prepare_data.py --zips-dir "caminho/para/zips"
```

### Treinar Modelo
```bash
python scripts/train.py --epochs 300 --batch-size 16
```

### Validar Modelo
```bash
python scripts/validate.py --weights pesos/best.pt --verbose
```

### Detectar em Imagens
```bash
python scripts/detect.py --weights pesos/best.pt --source imagens/
```

### Usar como Módulo Python
```python
from src.training import YOLOv9Trainer

trainer = YOLOv9Trainer()
trainer.train(epochs=300)
```

---

## 📁 Estrutura Rápida

```
src/                    # 🎯 Código modular
├── config.py          # Configurações
├── utils.py           # Utilitários
├── data_preparation.py # Preparação
├── training.py        # Treinamento
├── validation.py      # Validação
└── detection.py       # Detecção

scripts/               # 📜 Scripts CLI
├── prepare_data.py
├── train.py
├── validate.py
└── detect.py

docs/                  # 📚 Documentação
├── EXPLICACAO_ARQUITETURA.md
├── EXEMPLOS_PRATICOS.md
├── UV_GUIDE.md
├── API_REFERENCE.md
├── TECHNICAL_DOCUMENTATION.md
└── SETUP_GUIDE.md
```

---

## ❓ Perguntas Frequentes

### P: Qual é a diferença entre `src/` e `scripts/`?
**R**: 
- `src/` = Código reutilizável (módulos)
- `scripts/` = Interface CLI (scripts)

Leia: [docs/EXPLICACAO_ARQUITETURA.md](docs/EXPLICACAO_ARQUITETURA.md)

### P: Como uso como módulo Python?
**R**: 
```python
from src.training import YOLOv9Trainer
trainer = YOLOv9Trainer()
trainer.train(epochs=300)
```

Leia: [docs/EXEMPLOS_PRATICOS.md](docs/EXEMPLOS_PRATICOS.md)

### P: O que é `uv`?
**R**: Um gerenciador de pacotes Python 10-100x mais rápido que `pip`.

Leia: [docs/UV_GUIDE.md](docs/UV_GUIDE.md)

### P: Como instalo?
**R**: 
```bash
uv sync
source .venv/bin/activate
```

Leia: [QUICK_START.md](QUICK_START.md)

### P: Posso usar `pip` ao invés de `uv`?
**R**: Sim! Use `pip install -r requirements_refatorado.txt`

Leia: [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md)

---

## 🎓 Conceitos-Chave

### Módulo vs Script
- **Módulo** (`src/`): Código importado por outro
- **Script** (`scripts/`): Código executado diretamente

### Type Hints
```python
def validate(self, weights: str, img_size: int = None) -> int:
```

### Docstrings
```python
"""
Executa validação do modelo.

Args:
    weights: Caminho para os pesos
    
Returns:
    Código de retorno
"""
```

### Logging
```python
logger = setup_logging()
logger.info("Iniciando...")
```

---

## ✅ Checklist de Primeiro Uso

- [ ] Ler [QUICK_START.md](QUICK_START.md)
- [ ] Executar `uv sync`
- [ ] Ativar ambiente virtual
- [ ] Ler [docs/EXPLICACAO_ARQUITETURA.md](docs/EXPLICACAO_ARQUITETURA.md)
- [ ] Ver [docs/EXEMPLOS_PRATICOS.md](docs/EXEMPLOS_PRATICOS.md)
- [ ] Executar `python scripts/train.py --help`
- [ ] Começar a usar!

---

## 🚀 Próximo Passo

**Leia [QUICK_START.md](QUICK_START.md) agora!** ⏱️

---

## 📞 Precisa de Ajuda?

| Problema | Solução |
|----------|---------|
| Não entendo a estrutura | Leia [docs/EXPLICACAO_ARQUITETURA.md](docs/EXPLICACAO_ARQUITETURA.md) |
| Quero ver exemplos | Leia [docs/EXEMPLOS_PRATICOS.md](docs/EXEMPLOS_PRATICOS.md) |
| Erro de instalação | Leia [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md) |
| Não sei usar `uv` | Leia [docs/UV_GUIDE.md](docs/UV_GUIDE.md) |
| Preciso de referência da API | Leia [docs/API_REFERENCE.md](docs/API_REFERENCE.md) |

---

**Bem-vindo ao projeto refatorado! Aproveite! 🎉**
