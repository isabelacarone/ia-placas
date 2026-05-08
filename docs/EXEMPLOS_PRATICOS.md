# Exemplos Práticos - Como Usar os Módulos

## 📚 Índice
1. [Usar como Script (CLI)](#usar-como-script-cli)
2. [Usar como Módulo (Programático)](#usar-como-módulo-programático)
3. [Exemplos Completos](#exemplos-completos)
4. [Casos de Uso Reais](#casos-de-uso-reais)

---

## 🖥️ Usar como Script (CLI)

### O que é?
Executar o código via linha de comando, como um programa normal.

### Exemplo 1: Validação Básica

```bash
python scripts/validate.py --weights pesos/best.pt
```

**O que acontece**:
1. Script `scripts/validate.py` é executado
2. Importa `YOLOv9Validator` de `src/validation.py`
3. Cria instância e chama `validate()`
4. Mostra resultados

**Output esperado**:
```
============================================================
                VALIDAÇÃO YOLOv9 - Detector de Placas
============================================================
  Pesos:      pesos/best.pt
  Confianca:  0.001
  IoU:        0.7
  Device:     0
============================================================
Iniciando validação...
Validação concluída com sucesso!

Resultados salvos em: uvv/yolov9-main/runs/val/
```

### Exemplo 2: Validação com Opções Customizadas

```bash
python scripts/validate.py \
    --weights pesos/best.pt \
    --img-size 416 \
    --batch-size 16 \
    --device cpu \
    --verbose
```

**Parâmetros**:
- `--weights`: Caminho para os pesos
- `--img-size`: Tamanho da imagem (padrão: 640)
- `--batch-size`: Tamanho do batch (padrão: 32)
- `--device`: GPU (0) ou CPU (padrão: 0)
- `--verbose`: Mostrar métricas por classe

### Exemplo 3: Treinamento

```bash
# Treinamento básico
python scripts/train.py

# Treinamento customizado
python scripts/train.py \
    --epochs 500 \
    --batch-size 16 \
    --device 0 \
    --name meu_detector

# Retomar treinamento
python scripts/train.py \
    --resume uvv/yolov9-main/runs/train/detector_placas/weights/last.pt
```

### Exemplo 4: Detecção

```bash
# Detecção em pasta
python scripts/detect.py \
    --weights pesos/best.pt \
    --source imagens/

# Detecção em imagem específica
python scripts/detect.py \
    --weights pesos/best.pt \
    --source imagem.jpg \
    --conf 0.5 \
    --save-txt

# Detecção com configurações avançadas
python scripts/detect.py \
    --weights pesos/best.pt \
    --source imagens/ \
    --conf 0.6 \
    --iou 0.4 \
    --device cpu \
    --save-txt \
    --save-conf
```

---

## 🐍 Usar como Módulo (Programático)

### O que é?
Importar o código em seus próprios scripts Python e usar como biblioteca.

### Exemplo 1: Validação Simples

```python
from src.validation import YOLOv9Validator

# Criar instância
validator = YOLOv9Validator()

# Executar validação
return_code = validator.validate(
    weights="pesos/best.pt",
    verbose=True
)

# Verificar resultado
if return_code == 0:
    print("✓ Validação bem-sucedida!")
else:
    print("✗ Validação falhou!")
```

### Exemplo 2: Treinamento com Monitoramento

```python
from src.training import YOLOv9Trainer
from src.utils import setup_logging

# Configurar logging
logger = setup_logging()

# Criar treinador
trainer = YOLOv9Trainer()

# Treinar modelo
logger.info("Iniciando treinamento...")
return_code = trainer.train(
    epochs=300,
    batch_size=16,
    device="0",
    name="detector_v2"
)

# Processar resultado
if return_code == 0:
    logger.info("Treinamento concluído!")
    # Fazer algo com o modelo treinado
else:
    logger.error("Treinamento falhou!")
```

### Exemplo 3: Detecção e Processamento

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
    print(f"{image_name}: {data['plate_text']} (confiança: {data['confidence']:.2f})")
```

### Exemplo 4: Preparação de Dados

```python
from src.data_preparation import DataPreparator
from pathlib import Path

# Criar preparador
preparator = DataPreparator()

# Extrair dados
preparator.extract_datasets(Path("zips/"))

# Validar estrutura
stats = preparator.validate_dataset_structure()

# Exibir estatísticas
for split_name, stats in stats.items():
    print(f"{split_name}: {stats['images']} imagens, {stats['labels']} labels")

# Criar arquivo YAML
yaml_path = preparator.create_data_yaml()
print(f"Configuração salva em: {yaml_path}")
```

### Exemplo 5: Usar Configurações

```python
from src.config import config

# Acessar configurações
print(f"Número de classes: {config.NUM_CLASSES}")
print(f"Classes: {config.CLASS_NAMES}")
print(f"Épocas padrão: {config.DEFAULT_EPOCHS}")
print(f"Batch size padrão: {config.DEFAULT_BATCH_SIZE}")

# Validar caminhos
paths_valid = config.validate_paths()
for name, exists in paths_valid.items():
    status = "✓" if exists else "✗"
    print(f"{status} {name}")

# Criar diretórios necessários
config.create_directories()
```

### Exemplo 6: Usar Utilitários

```python
from src.utils import (
    setup_logging,
    validate_device,
    check_yolov9_installation,
    get_class_mapping,
    parse_yolo_label
)
from pathlib import Path

# Configurar logging
logger = setup_logging()
logger.info("Iniciando aplicação...")

# Validar dispositivo
device = validate_device("0")
logger.info(f"Usando dispositivo: {device}")

# Verificar instalação
if check_yolov9_installation():
    logger.info("✓ YOLOv9 instalado corretamente")
else:
    logger.error("✗ YOLOv9 não encontrado")

# Obter mapeamento de classes
class_mapping = get_class_mapping()
print(f"Classe 0: {class_mapping[0]}")  # '0'
print(f"Classe 10: {class_mapping[10]}")  # 'A'

# Parsear arquivo de label
detections = parse_yolo_label(Path("label.txt"))
for det in detections:
    print(f"Classe: {det['class_name']}, Confiança: {det['confidence']}")
```

---

## 🎯 Exemplos Completos

### Exemplo 1: Pipeline Completo

```python
"""
Pipeline completo: Preparar dados → Treinar → Validar → Detectar
"""

from src.data_preparation import prepare_data_from_zips
from src.training import YOLOv9Trainer
from src.validation import YOLOv9Validator
from src.detection import YOLOv9Detector, PlateCharacterProcessor
from src.utils import setup_logging
from pathlib import Path

# Configurar logging
logger = setup_logging()

# 1. PREPARAR DADOS
logger.info("=== ETAPA 1: Preparando dados ===")
prepare_data_from_zips(
    zips_dir="zips/",
    data_dir="dados"
)

# 2. TREINAR MODELO
logger.info("=== ETAPA 2: Treinando modelo ===")
trainer = YOLOv9Trainer()
train_code = trainer.train(
    epochs=300,
    batch_size=16,
    device="0",
    name="detector_v1"
)

if train_code != 0:
    logger.error("Treinamento falhou!")
    exit(1)

# 3. VALIDAR MODELO
logger.info("=== ETAPA 3: Validando modelo ===")
validator = YOLOv9Validator()
val_code = validator.validate(
    weights="uvv/yolov9-main/runs/train/detector_v1/weights/best.pt",
    verbose=True
)

if val_code != 0:
    logger.error("Validação falhou!")
    exit(1)

# 4. EXECUTAR DETECÇÃO
logger.info("=== ETAPA 4: Executando detecção ===")
detector = YOLOv9Detector()
detect_code = detector.detect(
    weights="uvv/yolov9-main/runs/train/detector_v1/weights/best.pt",
    source="imagens_teste/",
    save_txt=True,
    save_conf=True
)

# 5. PROCESSAR RESULTADOS
logger.info("=== ETAPA 5: Processando resultados ===")
processor = PlateCharacterProcessor()
results = processor.process_detection_results(
    results_dir=Path("uvv/yolov9-main/runs/detect/deteccao_placas/"),
    confidence_threshold=0.6
)

# 6. EXIBIR RESULTADOS
logger.info("=== RESULTADOS FINAIS ===")
for image_name, data in results.items():
    print(f"{image_name}:")
    print(f"  Placa: {data['plate_text']}")
    print(f"  Confiança: {data['confidence']:.2f}")
    print(f"  Detecções: {len(data['detections'])}")

logger.info("✓ Pipeline completo concluído!")
```

### Exemplo 2: Validação Automática de Múltiplos Modelos

```python
"""
Validar múltiplos modelos e comparar resultados
"""

from src.validation import YOLOv9Validator
from pathlib import Path
import json

# Lista de modelos para validar
models = [
    "pesos/best_v1.pt",
    "pesos/best_v2.pt",
    "pesos/best_v3.pt",
]

# Validar cada modelo
validator = YOLOv9Validator()
results = {}

for model_path in models:
    model_name = Path(model_path).stem
    print(f"\nValidando {model_name}...")
    
    return_code = validator.validate(
        weights=model_path,
        verbose=True,
        name=f"val_{model_name}"
    )
    
    results[model_name] = {
        "path": model_path,
        "success": return_code == 0
    }

# Salvar resultados
with open("validation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✓ Validação de múltiplos modelos concluída!")
```

### Exemplo 3: Detecção em Tempo Real (Simulado)

```python
"""
Simular detecção em tempo real processando imagens de uma pasta
"""

from src.detection import YOLOv9Detector, PlateCharacterProcessor
from pathlib import Path
import time

# Configurar detector
detector = YOLOv9Detector()
processor = PlateCharacterProcessor()

# Pasta com imagens
images_dir = Path("imagens_tempo_real/")

# Processar imagens continuamente
print("Iniciando detecção em tempo real...")
print("Pressione Ctrl+C para parar\n")

try:
    while True:
        # Listar imagens
        images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        
        if not images:
            print("Nenhuma imagem encontrada. Aguardando...")
            time.sleep(5)
            continue
        
        # Processar cada imagem
        for image_path in images:
            print(f"Processando: {image_path.name}")
            
            # Executar detecção
            detector.detect(
                weights="pesos/best.pt",
                source=str(image_path),
                save_txt=True
            )
            
            # Processar resultados
            results = processor.process_detection_results(
                results_dir=Path("uvv/yolov9-main/runs/detect/deteccao_placas/")
            )
            
            # Exibir resultado
            for img_name, data in results.items():
                print(f"  Placa detectada: {data['plate_text']}")
            
            # Mover imagem processada
            image_path.rename(images_dir / f"processada_{image_path.name}")
        
        # Aguardar novas imagens
        time.sleep(5)

except KeyboardInterrupt:
    print("\n✓ Detecção em tempo real finalizada!")
```

---

## 🔧 Casos de Uso Reais

### Caso 1: Integração com API Web

```python
"""
Integrar detecção com API Flask
"""

from flask import Flask, request, jsonify
from src.detection import YOLOv9Detector, PlateCharacterProcessor
from pathlib import Path
import tempfile

app = Flask(__name__)
detector = YOLOv9Detector()
processor = PlateCharacterProcessor()

@app.route('/detect', methods=['POST'])
def detect_plate():
    """Endpoint para detectar placa em imagem"""
    
    # Receber imagem
    if 'image' not in request.files:
        return jsonify({"error": "Nenhuma imagem fornecida"}), 400
    
    image_file = request.files['image']
    
    # Salvar temporariamente
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        image_file.save(tmp.name)
        temp_path = tmp.name
    
    try:
        # Executar detecção
        detector.detect(
            weights="pesos/best.pt",
            source=temp_path,
            save_txt=True
        )
        
        # Processar resultados
        results = processor.process_detection_results(
            results_dir=Path("uvv/yolov9-main/runs/detect/deteccao_placas/")
        )
        
        # Retornar resultado
        for img_name, data in results.items():
            return jsonify({
                "plate": data['plate_text'],
                "confidence": data['confidence'],
                "detections": len(data['detections'])
            })
        
        return jsonify({"error": "Nenhuma placa detectada"}), 404
        
    finally:
        # Limpar arquivo temporário
        Path(temp_path).unlink()

if __name__ == '__main__':
    app.run(debug=True)
```

### Caso 2: Batch Processing

```python
"""
Processar múltiplas imagens em lote
"""

from src.detection import YOLOv9Detector, PlateCharacterProcessor
from pathlib import Path
import csv

# Configurar
detector = YOLOv9Detector()
processor = PlateCharacterProcessor()
images_dir = Path("imagens_lote/")

# Processar todas as imagens
detector.detect(
    weights="pesos/best.pt",
    source=str(images_dir),
    save_txt=True,
    save_conf=True
)

# Processar resultados
results = processor.process_detection_results(
    results_dir=Path("uvv/yolov9-main/runs/detect/deteccao_placas/"),
    confidence_threshold=0.5
)

# Salvar em CSV
with open("resultados.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Imagem", "Placa", "Confiança", "Num Caracteres"])
    
    for image_name, data in results.items():
        writer.writerow([
            image_name,
            data['plate_text'],
            f"{data['confidence']:.2f}",
            len(data['detections'])
        ])

print("✓ Resultados salvos em resultados.csv")
```

### Caso 3: Monitoramento de Performance

```python
"""
Monitorar performance do modelo durante validação
"""

from src.validation import YOLOv9Validator
from src.utils import setup_logging
import time

logger = setup_logging()

# Validar modelo
validator = YOLOv9Validator()

# Medir tempo
start_time = time.time()

return_code = validator.validate(
    weights="pesos/best.pt",
    batch_size=32,
    verbose=True
)

elapsed_time = time.time() - start_time

# Exibir estatísticas
logger.info(f"Tempo de validação: {elapsed_time:.2f}s")
logger.info(f"Status: {'✓ Sucesso' if return_code == 0 else '✗ Falha'}")

# Salvar estatísticas
with open("performance.txt", "a") as f:
    f.write(f"Validação: {elapsed_time:.2f}s - {return_code}\n")
```

---

## 📝 Resumo

| Tipo | Quando Usar | Exemplo |
|------|------------|---------|
| **Script CLI** | Uso simples, linha de comando | `python scripts/validate.py --weights pesos/best.pt` |
| **Módulo Python** | Integração, automação | `from src.validation import YOLOv9Validator` |
| **Pipeline** | Fluxo completo | Preparar → Treinar → Validar → Detectar |
| **API Web** | Serviço online | Flask, FastAPI |
| **Batch** | Processar múltiplas imagens | CSV, JSON |

Agora você sabe como usar cada parte do projeto! 🚀