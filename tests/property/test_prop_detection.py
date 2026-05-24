"""
Testes de propriedade para módulo detection usando Hypothesis.

Feature: detector-placas-yolov9
Properties:
- Property 4: filtragem por confidence_threshold é correta e completa
- Property 5: reconstrução de texto ordena por x_center
- Property 6: class_mapping é bijetivo e cobre todos os 35 índices

Valida: Requisitos 7.2, 7.3, 7.8
"""

import pytest
from unittest.mock import patch, MagicMock
from hypothesis import given, settings, strategies as st, assume
from src.detection import PlateCharacterProcessor
from src.config import config


# Estratégia para gerar detecções válidas
@st.composite
def detection_strategy(draw):
    """
    Gera um dicionário de detecção válido com todos os campos necessários.
    
    Campos:
    - class_id: int 0-34
    - class_name: str (um dos 35 caracteres válidos)
    - x_center, y_center, width, height: float 0.0-1.0
    - confidence: float 0.0-1.0
    """
    class_id = draw(st.integers(min_value=0, max_value=34))
    # Usar os nomes de classe reais do config
    class_name = config.CLASS_NAMES[class_id]
    
    return {
        'class_id': class_id,
        'class_name': class_name,
        'x_center': draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        'y_center': draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        'width': draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        'height': draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        'confidence': draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    }


# Feature: detector-placas-yolov9, Property 4: filtragem por confidence_threshold é correta e completa
@given(
    st.lists(detection_strategy(), min_size=0, max_size=20),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=200)
def test_filter_by_confidence_correct_and_complete(detections, threshold):
    """
    Property 4: Para qualquer lista de detecções e threshold em [0.0, 1.0],
    _filter_by_confidence retorna apenas detecções com confidence >= threshold
    e não descarta nenhuma que deveria passar.
    
    Esta propriedade garante que:
    1. Todas as detecções retornadas têm confidence >= threshold (corretude)
    2. Nenhuma detecção com confidence >= threshold é descartada (completude)
    """
    processor = PlateCharacterProcessor()
    
    # Filtrar detecções
    filtered = processor._filter_by_confidence(detections, threshold)
    
    # Verificar corretude: todas as detecções retornadas devem ter confidence >= threshold
    for detection in filtered:
        assert detection['confidence'] >= threshold, \
            f"Detecção com confidence {detection['confidence']} passou com threshold {threshold}"
    
    # Verificar completude: contar quantas detecções deveriam passar
    expected_count = sum(1 for d in detections if d['confidence'] >= threshold)
    assert len(filtered) == expected_count, \
        f"Esperado {expected_count} detecções, mas obteve {len(filtered)}"
    
    # Verificar que as detecções filtradas são um subconjunto das originais
    for detection in filtered:
        assert detection in detections, "Detecção filtrada não está na lista original"


@given(st.lists(detection_strategy(), min_size=1, max_size=20))
@settings(max_examples=200)
def test_filter_by_confidence_threshold_zero_returns_all(detections):
    """
    Property 4a: Threshold 0.0 deve retornar todas as detecções.
    
    Esta propriedade garante que o caso extremo de threshold mínimo
    funciona corretamente.
    """
    processor = PlateCharacterProcessor()
    filtered = processor._filter_by_confidence(detections, 0.0)
    
    assert len(filtered) == len(detections)


@given(st.lists(detection_strategy(), min_size=0, max_size=20))
@settings(max_examples=200)
def test_filter_by_confidence_threshold_one_returns_only_perfect(detections):
    """
    Property 4b: Threshold 1.0 deve retornar apenas detecções com confidence exatamente 1.0.
    
    Esta propriedade garante que o caso extremo de threshold máximo
    funciona corretamente.
    """
    processor = PlateCharacterProcessor()
    filtered = processor._filter_by_confidence(detections, 1.0)
    
    expected_count = sum(1 for d in detections if d['confidence'] >= 1.0)
    assert len(filtered) == expected_count
    
    for detection in filtered:
        assert detection['confidence'] >= 1.0


# Feature: detector-placas-yolov9, Property 5: reconstrução de texto ordena por x_center
@given(st.lists(detection_strategy(), min_size=1, max_size=10))
@settings(max_examples=200)
def test_reconstruct_plate_text_orders_by_x_center(detections):
    """
    Property 5: Para qualquer lista não-vazia de detecções,
    _reconstruct_plate_text retorna concatenação dos class_name
    na ordem crescente de x_center.
    
    Esta propriedade garante que os caracteres da placa são ordenados
    corretamente da esquerda para a direita.
    """
    processor = PlateCharacterProcessor()
    
    # Reconstruir texto
    text = processor._reconstruct_plate_text(detections)
    
    # Ordenar detecções manualmente por x_center
    sorted_detections = sorted(detections, key=lambda d: d['x_center'])
    expected_text = ''.join(d['class_name'] for d in sorted_detections)
    
    assert text == expected_text, \
        f"Texto reconstruído '{text}' não corresponde ao esperado '{expected_text}'"


@given(st.lists(detection_strategy(), min_size=2, max_size=10))
@settings(max_examples=200)
def test_reconstruct_plate_text_concatenates_without_separator(detections):
    """
    Property 5a: O texto reconstruído não contém separadores entre caracteres.
    
    Esta propriedade garante que os caracteres são concatenados diretamente,
    sem espaços ou outros separadores.
    """
    processor = PlateCharacterProcessor()
    text = processor._reconstruct_plate_text(detections)
    
    # Verificar que não há espaços
    assert ' ' not in text
    
    # Verificar que o comprimento corresponde ao número de detecções
    assert len(text) == len(detections)


@given(detection_strategy())
@settings(max_examples=200)
def test_reconstruct_plate_text_single_detection(detection):
    """
    Property 5b: Uma única detecção retorna apenas seu class_name.
    
    Esta propriedade garante que o caso base funciona corretamente.
    """
    processor = PlateCharacterProcessor()
    text = processor._reconstruct_plate_text([detection])
    
    assert text == detection['class_name']


# Feature: detector-placas-yolov9, Property 6: class_mapping é bijetivo e cobre todos os 35 índices
def test_class_mapping_has_35_entries():
    """
    Property 6a: class_mapping tem exatamente 35 entradas.
    
    Esta propriedade garante que o mapeamento cobre todas as classes
    de caracteres definidas no sistema.
    """
    processor = PlateCharacterProcessor()
    
    assert len(processor.class_mapping) == 35


def test_class_mapping_keys_are_0_to_34():
    """
    Property 6b: class_mapping tem chaves 0–34.
    
    Esta propriedade garante que todos os índices de classe válidos
    estão presentes no mapeamento.
    """
    processor = PlateCharacterProcessor()
    
    assert set(processor.class_mapping.keys()) == set(range(35))


def test_class_mapping_values_match_config_class_names():
    """
    Property 6c: Valores de class_mapping são idênticos a config.CLASS_NAMES.
    
    Esta propriedade garante que o mapeamento está sincronizado com
    a configuração global do sistema.
    """
    processor = PlateCharacterProcessor()
    
    for i in range(35):
        assert processor.class_mapping[i] == config.CLASS_NAMES[i], \
            f"Mapeamento incorreto no índice {i}: esperado '{config.CLASS_NAMES[i]}', obteve '{processor.class_mapping[i]}'"


def test_class_mapping_is_bijective():
    """
    Property 6d: class_mapping é bijetivo (cada valor aparece exatamente uma vez).
    
    Esta propriedade garante que não há duplicatas nos nomes de classe,
    permitindo mapeamento reverso único.
    """
    processor = PlateCharacterProcessor()
    
    values = list(processor.class_mapping.values())
    assert len(values) == len(set(values)), "class_mapping contém valores duplicados"


def test_class_mapping_contains_all_expected_characters():
    """
    Property 6e: class_mapping contém todos os caracteres esperados.
    
    Esta propriedade garante que o mapeamento inclui:
    - Dígitos 0-9 (10 caracteres)
    - Letras A-Z exceto O (25 caracteres)
    Total: 35 caracteres
    """
    processor = PlateCharacterProcessor()
    
    values = set(processor.class_mapping.values())
    
    # Verificar dígitos 0-9
    for digit in '0123456789':
        assert digit in values, f"Dígito '{digit}' não encontrado no mapeamento"
    
    # Verificar letras A-Z exceto O
    for letter in 'ABCDEFGHIJKLMNPQRSTUVWXYZ':  # Note: sem O
        assert letter in values, f"Letra '{letter}' não encontrada no mapeamento"
    
    # Verificar que O não está presente
    assert 'O' not in values, "Letra 'O' não deveria estar no mapeamento"


# Teste de invariante: operações com PlateCharacterProcessor são determinísticas
@given(st.lists(detection_strategy(), min_size=1, max_size=10))
@settings(max_examples=200)
def test_reconstruct_plate_text_is_deterministic(detections):
    """
    Invariante: _reconstruct_plate_text é determinístico.
    
    Esta propriedade garante que múltiplas chamadas com os mesmos dados
    produzem o mesmo resultado.
    """
    processor = PlateCharacterProcessor()
    
    text1 = processor._reconstruct_plate_text(detections)
    text2 = processor._reconstruct_plate_text(detections)
    
    assert text1 == text2


@given(
    st.lists(detection_strategy(), min_size=0, max_size=20),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=200)
def test_filter_by_confidence_is_deterministic(detections, threshold):
    """
    Invariante: _filter_by_confidence é determinístico.
    
    Esta propriedade garante que múltiplas chamadas com os mesmos dados
    produzem o mesmo resultado.
    """
    processor = PlateCharacterProcessor()
    
    filtered1 = processor._filter_by_confidence(detections, threshold)
    filtered2 = processor._filter_by_confidence(detections, threshold)
    
    assert filtered1 == filtered2
