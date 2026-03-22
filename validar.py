"""
Script de validacao do modelo YOLOv9 treinado.

Uso:
    python validar.py --weights pesos/best.pt
    python validar.py --weights pesos/best.pt --conf 0.2 --device cpu
"""

import os
import sys
import subprocess
from pathlib import Path

RAIZ = Path(__file__).parent
YOLOV9_DIR = RAIZ / "uvv" / "yolov9-main"

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"


def validar():
    import argparse

    parser = argparse.ArgumentParser(description="Validar modelo YOLOv9 de placas")
    parser.add_argument("--weights", type=str, required=True, help="Caminho para os pesos do modelo (.pt)")
    parser.add_argument("--img-size", type=int, default=640, help="Tamanho da imagem (padrao: 640)")
    parser.add_argument("--device", type=str, default="0", help="Device: 0 para GPU, cpu para CPU")
    parser.add_argument("--conf", type=float, default=0.001, help="Threshold de confianca (padrao: 0.001)")
    parser.add_argument("--iou", type=float, default=0.7, help="Threshold de IoU (padrao: 0.7)")
    parser.add_argument("--name", type=str, default="validacao_placas", help="Nome do experimento")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamanho do batch (padrao: 32)")
    parser.add_argument("--verbose", action="store_true", help="Mostrar metricas por classe")
    args = parser.parse_args()

    data_yaml = RAIZ / "configs" / "dados-placas.yaml"
    val_script = YOLOV9_DIR / "val.py"

    if not val_script.exists():
        print(f"[ERRO] Script de validacao nao encontrado: {val_script}")
        sys.exit(1)

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"[ERRO] Pesos nao encontrados: {weights_path}")
        sys.exit(1)

    cmd = [
        sys.executable, str(val_script),
        "--data", str(data_yaml),
        "--weights", str(weights_path),
        "--imgsz", str(args.img_size),
        "--device", args.device,
        "--conf-thres", str(args.conf),
        "--iou-thres", str(args.iou),
        "--name", args.name,
        "--batch-size", str(args.batch_size),
    ]

    if args.verbose:
        cmd.append("--verbose")

    print("=" * 60)
    print("VALIDACAO YOLOv9 - Detector de Placas")
    print("=" * 60)
    print(f"  Pesos:      {weights_path}")
    print(f"  Confianca:  {args.conf}")
    print(f"  IoU:        {args.iou}")
    print(f"  Device:     {args.device}")
    print("=" * 60)

    result = subprocess.run(cmd, cwd=str(YOLOV9_DIR))
    sys.exit(result.returncode)


if __name__ == "__main__":
    validar()
