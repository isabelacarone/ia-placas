"""
Script de deteccao/inferencia com YOLOv9 em imagens de placas.

Uso:
    python detectar.py --weights pesos/best.pt --source imagens/
    python detectar.py --weights pesos/best.pt --source imagens/placa.jpg
    python detectar.py --weights pesos/best.pt --source imagens/ --device cpu --conf 0.25
"""

import os
import sys
import subprocess
from pathlib import Path

RAIZ = Path(__file__).parent
YOLOV9_DIR = RAIZ / "uvv" / "yolov9-main"


def detectar():
    import argparse

    parser = argparse.ArgumentParser(description="Detectar caracteres em placas com YOLOv9")
    parser.add_argument("--weights", type=str, required=True, help="Caminho para os pesos do modelo (.pt)")
    parser.add_argument("--source", type=str, required=True, help="Imagem, pasta de imagens, ou video")
    parser.add_argument("--img-size", type=int, default=640, help="Tamanho da imagem (padrao: 640)")
    parser.add_argument("--device", type=str, default="0", help="Device: 0 para GPU, cpu para CPU")
    parser.add_argument("--conf", type=float, default=0.25, help="Threshold de confianca (padrao: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45, help="Threshold de IoU NMS (padrao: 0.45)")
    parser.add_argument("--name", type=str, default="deteccao_placas", help="Nome do experimento")
    parser.add_argument("--save-txt", action="store_true", help="Salvar resultados em .txt")
    parser.add_argument("--save-conf", action="store_true", help="Salvar confiancas nos .txt")
    args = parser.parse_args()

    detect_script = YOLOV9_DIR / "detect.py"

    if not detect_script.exists():
        print(f"[ERRO] Script de deteccao nao encontrado: {detect_script}")
        sys.exit(1)

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"[ERRO] Pesos nao encontrados: {weights_path}")
        sys.exit(1)

    source_path = Path(args.source)
    if not source_path.exists():
        print(f"[ERRO] Source nao encontrado: {source_path}")
        sys.exit(1)

    cmd = [
        sys.executable, str(detect_script),
        "--weights", str(weights_path),
        "--source", str(source_path),
        "--img-size", str(args.img_size),
        "--device", args.device,
        "--conf-thres", str(args.conf),
        "--iou-thres", str(args.iou),
        "--name", args.name,
    ]

    if args.save_txt:
        cmd.append("--save-txt")
    if args.save_conf:
        cmd.append("--save-conf")

    print("=" * 60)
    print("DETECCAO YOLOv9 - Detector de Placas")
    print("=" * 60)
    print(f"  Pesos:      {weights_path}")
    print(f"  Source:     {source_path}")
    print(f"  Confianca:  {args.conf}")
    print(f"  Device:     {args.device}")
    print("=" * 60)

    result = subprocess.run(cmd, cwd=str(YOLOV9_DIR))

    if result.returncode == 0:
        print("\nResultados salvos em: uvv/yolov9-main/runs/detect/")

    sys.exit(result.returncode)


if __name__ == "__main__":
    detectar()
