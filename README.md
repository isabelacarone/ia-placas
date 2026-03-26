# Detector de Caracteres em Placas Veiculares - YOLOv9

Projeto de detecção de caracteres individuais em placas veiculares brasileiras usando YOLOv9.

## Estrutura do Projeto

```
trabalho-ia/
├── configs/
│   ├── dados-placas.yaml       # Configuração do dataset (classes, caminhos)
│   └── hyp.scratch-high.yaml   # Hiperparâmetros de treinamento
├── dados/
│   ├── treino/                 # Imagens e labels de treino
│   ├── validacao/              # Imagens e labels de validação
│   └── teste/                  # Imagens e labels de teste
├── pesos/                      # Pesos do modelo treinado (.pt)
├── uvv/
│   └── yolov9-main/            # Código fonte do YOLOv9
├── preparar_dados.py           # Script para descompactar e organizar dados
├── treinar.py                  # Script de treinamento
├── validar.py                  # Script de validação
├── detectar.py                 # Script de detecção/inferência
└── requirements.txt            # Dependências do projeto
```

## Setup Inicial

### 1. Criar ambiente virtual

```bash
python -m venv venv
source venv/bin/activate   
# venv\Scripts\activate        # Windows
```

### 2. Instalar dependências

```bash
# com GPU NVIDIA - cuda 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu12

# Restante das dependências:
pip install -r requirements.txt
```

### 3. Preparar dados

Coloque os arquivos `Treino.zip`, `Validacao.zip` e `Teste.zip` em uma pasta e execute:

```bash
python preparar_dados.py --zips-dir "C:/caminho/para/os/zips"
```

Ou copie/insira as imagens e labels em `dados/treino/`, `dados/validacao/` e `dados/teste/`.

## Como Usar

### Treinar

```bash
# Treinar com GPU (padrão)
python treinar.py

# Configurar épocas e batch size
python treinar.py --epochs 100 --batch-size 4 --device 0

# Retomar treino interrompido
python treinar.py --resume uvv/yolov9-main/runs/train/detector_placas/weights/last.pt
```

### Validar

```bash
python validar.py --weights uvv/yolov9-main/runs/train/detector_placas/weights/best.pt
python validar.py --weights pesos/best.pt --verbose --device cpu
```

### Detectar

```bash
# Detectar em uma pasta de imagens
python detectar.py --weights pesos/best.pt --source imagens/

# Detectar em uma imagem específica
python detectar.py --weights pesos/best.pt --source imagens/placa.jpg --conf 0.25

# Salvar resultados em texto
python detectar.py --weights pesos/best.pt --source imagens/ --save-txt
```

Os resultados ficam salvos em `uvv/yolov9-main/runs/detect/`.

## Classes Detectadas (35)

Dígitos: `0 1 2 3 4 5 6 7 8 9`


Letras: `A B C D E F G H I J K L M N P Q R S T U V W Y Z`

## Rotulações

É importante ressaltar que não há o upload do meu dataset nesse repositório, mas contém o label utilizado e os caracteres que o modelo aceita no tópico anterior. 


O site utilziado para realizar a rotulação foi: https://www.makesense.ai/


Utilizando o **bounding box**
