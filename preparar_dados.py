"""
Script para preparar os dados do dataset de placas.

Descompacta os arquivos .zip de treino, validacao e teste
nas pastas corretas do projeto.

Uso:
    python preparar_dados.py --zips-dir "C:/caminho/para/os/zips"

Estrutura esperada dos ZIPs:
    - Treino.zip    -> dados/treino/
    - Validacao.zip -> dados/validacao/
    - Teste.zip     -> dados/teste/

Cada pasta deve conter:
    - images/ (imagens .jpg/.png)
    - labels/ (anotacoes .txt no formato YOLO)
"""

import argparse
import zipfile
import shutil
from pathlib import Path


def descompactar_dados(zips_dir: Path, dados_dir: Path):
    """Descompacta os ZIPs de treino, validacao e teste."""

    mapeamento = {
        "Treino.zip": dados_dir / "treino",
        "Validacao.zip": dados_dir / "validacao",
        "Teste.zip": dados_dir / "teste",
    }

    for nome_zip, pasta_destino in mapeamento.items():
        caminho_zip = zips_dir / nome_zip

        if not caminho_zip.exists():
            print(f"[AVISO] {caminho_zip} nao encontrado, pulando...")
            continue

        print(f"Descompactando {nome_zip} -> {pasta_destino}")
        pasta_destino.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(caminho_zip, "r") as zf:
            zf.extractall(pasta_destino)

        # YOLO espera a pasta "images" para mapear labels automaticamente.
        pasta_imagens = pasta_destino / "imagens"
        pasta_images = pasta_destino / "images"
        if pasta_imagens.exists() and not pasta_images.exists():
            pasta_imagens.rename(pasta_images)
            print(f"  [AJUSTE] Renomeado {pasta_imagens.name} -> {pasta_images.name}")

        print(f"  OK - {len(list(pasta_destino.rglob('*')))} arquivos extraidos")

    print("\nDados preparados com sucesso!")


def verificar_estrutura(dados_dir: Path):
    """Verifica se a estrutura de pastas esta correta."""
    print("\n--- Verificacao da estrutura ---")

    for split in ["treino", "validacao", "teste"]:
        pasta = dados_dir / split
        if not pasta.exists():
            print(f"[ERRO] Pasta {pasta} nao existe!")
            continue

        # Procura imagens em qualquer subpasta
        imagens = list(pasta.rglob("*.jpg")) + list(pasta.rglob("*.png"))
        labels = list(pasta.rglob("*.txt"))

        print(f"{split}: {len(imagens)} imagens, {len(labels)} labels")

        if len(imagens) == 0:
            print(f"  [AVISO] Nenhuma imagem encontrada em {pasta}")
        if len(labels) == 0:
            print(f"  [AVISO] Nenhum label encontrado em {pasta}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preparar dados do dataset de placas")
    parser.add_argument(
        "--zips-dir",
        type=str,
        required=True,
        help="Diretorio onde estao os arquivos ZIP (Treino.zip, Validacao.zip, Teste.zip)",
    )
    parser.add_argument(
        "--dados-dir",
        type=str,
        default="dados",
        help="Diretorio de destino dos dados (padrao: dados/)",
    )
    args = parser.parse_args()

    raiz = Path(__file__).parent
    zips_dir = Path(args.zips_dir)
    dados_dir = raiz / args.dados_dir

    descompactar_dados(zips_dir, dados_dir)
    verificar_estrutura(dados_dir)
