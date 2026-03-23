"""
Script de treinamento YOLOv9 para deteccao de caracteres em placas.

Uso basico:
    python treinar.py

Uso com GPU:
    python treinar.py --device 0

Uso com CPU:
    python treinar.py --device cpu

Retomar treino interrompido:
    python treinar.py --resume pesos/last.pt
"""

import os
import sys
import subprocess
from pathlib import Path

# Diretorio raiz do projeto
RAIZ = Path(__file__).parent
YOLOV9_DIR = RAIZ / "uvv" / "yolov9-main"

# Desabilitar WandB (evita pedir login)
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"


def treinar():
    import argparse

    parser = argparse.ArgumentParser(description="Treinar YOLOv9 para deteccao de placas")
    parser.add_argument("--epochs", type=int, default=300, help="Numero de epocas (padrao: 300)")
    parser.add_argument("--batch-size", type=int, default=8, help="Tamanho do batch (padrao: 8)")
    parser.add_argument("--img-size", type=int, default=640, help="Tamanho da imagem (padrao: 640)")
    parser.add_argument("--device", type=str, default="0", help="Device: 0 para GPU, cpu para CPU (padrao: 0)")
    parser.add_argument("--workers", type=int, default=0, help="Numero de workers para dataloader (padrao: 0)")
    parser.add_argument("--name", type=str, default="detector_placas", help="Nome do experimento")
    parser.add_argument("--resume", type=str, default="", help="Caminho para checkpoint para retomar treino")
    parser.add_argument("--weights", type=str, default="", help="Pesos pre-treinados (vazio = treinar do zero)")
    args = parser.parse_args()

    # Caminhos dos arquivos de configuracao
    data_yaml = RAIZ / "configs" / "dados-placas.yaml"
    model_yaml = YOLOV9_DIR / "models" / "detect" / "yolov9-c.yaml"
    hyp_yaml = RAIZ / "configs" / "hyp.scratch-high.yaml"
    train_script = YOLOV9_DIR / "train_dual.py"

    # Verificacoes
    if not train_script.exists():
        print(f"[ERRO] Script de treino nao encontrado: {train_script}")
        print("Verifique se a pasta uvv/yolov9-main/ existe.")
        sys.exit(1)

    if not data_yaml.exists():
        print(f"[ERRO] Arquivo de dados nao encontrado: {data_yaml}")
        sys.exit(1)

    # Montar comando de treino
    cmd = [
        sys.executable, str(train_script),
        "--workers", str(args.workers),
        "--device", args.device,
        "--batch-size", str(args.batch_size),
        "--data", str(data_yaml),
        "--img", str(args.img_size),
        "--cfg", str(model_yaml),
        "--weights", args.weights,
        "--name", args.name,
        "--hyp", str(hyp_yaml),
        "--epochs", str(args.epochs),
        "--noplots",
    ]

    if args.resume:
        cmd.extend(["--resume", args.resume])

    print("=" * 60)
    print("TREINAMENTO YOLOv9 - Detector de Placas")
    print("=" * 60)
    print(f"  Epocas:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Img size:   {args.img_size}")
    print(f"  Device:     {args.device}")
    print(f"  Nome:       {args.name}")
    print(f"  Data YAML:  {data_yaml}")
    print(f"  Model YAML: {model_yaml}")
    print("=" * 60)

    # Executar treino a partir do diretorio do yolov9
    result = subprocess.run(cmd, cwd=str(YOLOV9_DIR))
    sys.exit(result.returncode)


if __name__ == "__main__":
    treinar()
