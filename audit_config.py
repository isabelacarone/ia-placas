#!/usr/bin/env python3
"""
Script de auditoria para src/config.py

Verifica:
1. validate_paths retorna exatamente 7 chaves especificadas
2. CLASS_NAMES tem exatamente 35 elementos na ordem correta
3. get_timestamp() existe e funciona corretamente
"""

import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import config


def audit_validate_paths():
    """Audita o método validate_paths."""
    print("=" * 70)
    print("AUDITORIA 1: validate_paths()")
    print("=" * 70)
    
    expected_keys = {
        "YOLOv9 Directory",
        "Train Script",
        "Validation Script",
        "Detection Script",
        "Data YAML",
        "Hyperparams YAML",
        "Model YAML"
    }
    
    result = config.validate_paths()
    actual_keys = set(result.keys())
    
    print(f"\n✓ Chaves esperadas: {len(expected_keys)}")
    print(f"✓ Chaves retornadas: {len(actual_keys)}")
    
    if actual_keys == expected_keys:
        print("\n✅ PASSOU: validate_paths retorna exatamente as 7 chaves especificadas")
        print("\nChaves verificadas:")
        for key in sorted(expected_keys):
            status = "✓ existe" if result[key] else "✗ não existe"
            print(f"  - {key}: {status}")
        return True
    else:
        print("\n❌ FALHOU: Chaves não correspondem à especificação")
        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys
        if missing:
            print(f"\nChaves faltando: {missing}")
        if extra:
            print(f"\nChaves extras: {extra}")
        return False


def audit_class_names():
    """Audita CLASS_NAMES."""
    print("\n" + "=" * 70)
    print("AUDITORIA 2: CLASS_NAMES")
    print("=" * 70)
    
    # Verificar contagem
    expected_count = 35
    actual_count = len(config.CLASS_NAMES)
    
    print(f"\n✓ Elementos esperados: {expected_count}")
    print(f"✓ Elementos encontrados: {actual_count}")
    
    if actual_count != expected_count:
        print(f"\n❌ FALHOU: CLASS_NAMES tem {actual_count} elementos, esperado {expected_count}")
        return False
    
    # Verificar ordem: dígitos 0-9, depois letras A-Z exceto O
    expected_digits = [str(i) for i in range(10)]
    expected_letters = [chr(i) for i in range(ord('A'), ord('Z') + 1) if chr(i) != 'O']
    expected_classes = expected_digits + expected_letters
    
    if config.CLASS_NAMES == expected_classes:
        print("\n✅ PASSOU: CLASS_NAMES tem exatamente 35 elementos na ordem correta")
        print(f"\nDígitos (10): {expected_digits}")
        print(f"Letras (25): {expected_letters}")
        print(f"\n✓ Letra 'O' corretamente excluída")
        return True
    else:
        print("\n❌ FALHOU: CLASS_NAMES não está na ordem correta")
        print(f"\nEsperado: {expected_classes}")
        print(f"Atual:    {config.CLASS_NAMES}")
        return False


def audit_get_timestamp():
    """Audita o método get_timestamp."""
    print("\n" + "=" * 70)
    print("AUDITORIA 3: get_timestamp()")
    print("=" * 70)
    
    # Verificar se o método existe
    if not hasattr(config, 'get_timestamp'):
        print("\n❌ FALHOU: Método get_timestamp() não encontrado")
        return False
    
    # Verificar se é chamável
    if not callable(config.get_timestamp):
        print("\n❌ FALHOU: get_timestamp não é um método chamável")
        return False
    
    # Testar execução
    try:
        timestamp = config.get_timestamp()
        
        # Verificar se retorna string
        if not isinstance(timestamp, str):
            print(f"\n❌ FALHOU: get_timestamp() retorna {type(timestamp)}, esperado str")
            return False
        
        # Verificar formato (YYYYMMDD_HHMMSS)
        if len(timestamp) != 15 or timestamp[8] != '_':
            print(f"\n❌ FALHOU: Formato de timestamp inválido: {timestamp}")
            print("Esperado: YYYYMMDD_HHMMSS")
            return False
        
        print(f"\n✅ PASSOU: get_timestamp() existe e funciona corretamente")
        print(f"\nTimestamp gerado: {timestamp}")
        print(f"Formato: YYYYMMDD_HHMMSS")
        return True
        
    except Exception as e:
        print(f"\n❌ FALHOU: Erro ao executar get_timestamp(): {e}")
        return False


def audit_data_preparation_usage():
    """Verifica se data_preparation.py usa get_timestamp."""
    print("\n" + "=" * 70)
    print("AUDITORIA 4: Uso de get_timestamp() em data_preparation.py")
    print("=" * 70)
    
    data_prep_file = Path(__file__).parent / "src" / "data_preparation.py"
    
    if not data_prep_file.exists():
        print(f"\n❌ FALHOU: Arquivo não encontrado: {data_prep_file}")
        return False
    
    content = data_prep_file.read_text(encoding='utf-8')
    
    if 'config.get_timestamp()' in content:
        print("\n✅ PASSOU: data_preparation.py utiliza config.get_timestamp()")
        
        # Encontrar a linha
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if 'config.get_timestamp()' in line:
                print(f"\nLocalização: linha {i}")
                print(f"Contexto: {line.strip()}")
        return True
    else:
        print("\n❌ FALHOU: data_preparation.py não utiliza config.get_timestamp()")
        return False


def main():
    """Executa todas as auditorias."""
    print("\n" + "=" * 70)
    print("AUDITORIA DE src/config.py")
    print("Task 3.3: Verificar validate_paths, CLASS_NAMES e get_timestamp")
    print("=" * 70)
    
    results = []
    
    # Executar auditorias
    results.append(("validate_paths", audit_validate_paths()))
    results.append(("CLASS_NAMES", audit_class_names()))
    results.append(("get_timestamp", audit_get_timestamp()))
    results.append(("data_preparation usage", audit_data_preparation_usage()))
    
    # Resumo
    print("\n" + "=" * 70)
    print("RESUMO DA AUDITORIA")
    print("=" * 70)
    
    all_passed = all(result for _, result in results)
    
    for name, passed in results:
        status = "✅ PASSOU" if passed else "❌ FALHOU"
        print(f"{status}: {name}")
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ AUDITORIA CONCLUÍDA COM SUCESSO")
        print("Todos os requisitos estão conformes à especificação.")
        print("=" * 70)
        return 0
    else:
        print("❌ AUDITORIA FALHOU")
        print("Alguns requisitos não estão conformes à especificação.")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
