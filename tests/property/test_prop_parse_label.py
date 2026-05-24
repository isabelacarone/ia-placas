"""
Testes de propriedade para parse_yolo_label usando Hypothesis.

Feature: detector-placas-yolov9
Property:
- Property 3: parse_yolo_label preserva dados de linhas válidas

Valida: Requisito 2.5
"""

import pytest
from pathlib import Path
from hypothesis import given, settings, strategies as st, assume, HealthCheck
from src.utils import parse_yolo_label


# Estratégia para gerar linhas válidas de label YOLO (5 campos numéricos)
@st.composite
def valid_yolo_line(draw):
    """
    Gera uma linha válida de label YOLO com 5 campos numéricos.
    
    Formato: class_id x_center y_center width height
    - class_id: int 0-34
    - x_center, y_center, width, height: float 0.0-1.0
    """
    class_id = draw(st.integers(min_value=0, max_value=34))
    x_center = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    y_center = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    width = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    height = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    
    return f"{class_id} {x_center} {y_center} {width} {height}"


# Estratégia para gerar linhas inválidas (número de campos diferente de 5)
@st.composite
def invalid_yolo_line(draw):
    """
    Gera uma linha inválida de label YOLO com número de campos diferente de 5.
    """
    num_fields = draw(st.integers(min_value=0, max_value=10).filter(lambda x: x != 5))
    
    if num_fields == 0:
        return ""
    
    fields = []
    for _ in range(num_fields):
        field = draw(st.one_of(
            st.integers(min_value=0, max_value=100).map(str),
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False).map(str),
            st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll')), min_size=1, max_size=5)
        ))
        fields.append(field)
    
    return " ".join(fields)


# Feature: detector-placas-yolov9, Property 3: parse_yolo_label preserva dados de linhas válidas
@given(valid_lines=st.lists(valid_yolo_line(), min_size=0, max_size=20))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
def test_parse_yolo_label_preserves_valid_lines(valid_lines, tmp_path):
    """
    Property 3a: Para qualquer arquivo com N linhas de 5 campos numéricos,
    parse_yolo_label retorna lista com exatamente N dicts.
    
    Esta propriedade garante que todas as linhas válidas são processadas e
    nenhuma é perdida durante o parsing.
    """
    # Criar arquivo temporário com linhas válidas
    label_file = tmp_path / "test_label.txt"
    label_file.write_text("\n".join(valid_lines))
    
    # Parsear arquivo
    result = parse_yolo_label(label_file)
    
    # Verificar que o número de resultados corresponde ao número de linhas
    assert len(result) == len(valid_lines)
    
    # Verificar que cada resultado tem as chaves corretas
    for detection in result:
        assert set(detection.keys()) == {'class_id', 'x_center', 'y_center', 'width', 'height'}
        assert isinstance(detection['class_id'], int)
        assert isinstance(detection['x_center'], float)
        assert isinstance(detection['y_center'], float)
        assert isinstance(detection['width'], float)
        assert isinstance(detection['height'], float)


@given(
    valid_lines=st.lists(valid_yolo_line(), min_size=1, max_size=10),
    invalid_lines=st.lists(invalid_yolo_line(), min_size=1, max_size=10)
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
def test_parse_yolo_label_ignores_invalid_lines(valid_lines, invalid_lines, tmp_path):
    """
    Property 3b: Linhas com número de campos diferente de 5 são ignoradas.
    
    Esta propriedade garante que o parser é resiliente a dados malformados,
    processando apenas linhas válidas e ignorando silenciosamente as inválidas.
    """
    # Intercalar linhas válidas e inválidas
    all_lines = []
    for i in range(max(len(valid_lines), len(invalid_lines))):
        if i < len(valid_lines):
            all_lines.append(valid_lines[i])
        if i < len(invalid_lines):
            all_lines.append(invalid_lines[i])
    
    # Criar arquivo temporário
    label_file = tmp_path / "mixed_label.txt"
    label_file.write_text("\n".join(all_lines))
    
    # Parsear arquivo
    result = parse_yolo_label(label_file)
    
    # Verificar que apenas linhas válidas foram processadas
    assert len(result) == len(valid_lines)
    
    # Verificar estrutura dos resultados
    for detection in result:
        assert set(detection.keys()) == {'class_id', 'x_center', 'y_center', 'width', 'height'}


@given(valid_lines=st.lists(valid_yolo_line(), min_size=1, max_size=10))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
def test_parse_yolo_label_preserves_data_values(valid_lines, tmp_path):
    """
    Property 3c: Os valores dos campos são preservados corretamente.
    
    Esta propriedade garante que os dados numéricos são convertidos corretamente
    para os tipos apropriados (int para class_id, float para coordenadas).
    """
    # Criar arquivo temporário
    label_file = tmp_path / "test_label.txt"
    label_file.write_text("\n".join(valid_lines))
    
    # Parsear arquivo
    result = parse_yolo_label(label_file)
    
    # Verificar que os valores correspondem aos originais
    for i, detection in enumerate(result):
        original_fields = valid_lines[i].split()
        
        assert detection['class_id'] == int(original_fields[0])
        assert abs(detection['x_center'] - float(original_fields[1])) < 1e-6
        assert abs(detection['y_center'] - float(original_fields[2])) < 1e-6
        assert abs(detection['width'] - float(original_fields[3])) < 1e-6
        assert abs(detection['height'] - float(original_fields[4])) < 1e-6


def test_parse_yolo_label_returns_empty_list_for_nonexistent_file(tmp_path):
    """
    Property 3d: Arquivo inexistente retorna lista vazia sem exceção.
    
    Esta propriedade garante que o parser é resiliente a arquivos ausentes,
    permitindo processamento em lote sem interrupção.
    """
    nonexistent_file = tmp_path / "nonexistent.txt"
    result = parse_yolo_label(nonexistent_file)
    assert result == []
    assert isinstance(result, list)


def test_parse_yolo_label_returns_empty_list_for_empty_file(tmp_path):
    """
    Property 3e: Arquivo vazio retorna lista vazia.
    
    Esta propriedade garante que arquivos sem conteúdo são tratados
    corretamente, retornando uma lista vazia válida.
    """
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("")
    
    result = parse_yolo_label(empty_file)
    assert result == []
    assert isinstance(result, list)


# Teste de invariante: parse_yolo_label sempre retorna lista
@given(lines=st.lists(st.one_of(valid_yolo_line(), invalid_yolo_line()), min_size=0, max_size=20))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
def test_parse_yolo_label_always_returns_list(lines, tmp_path):
    """
    Invariante: parse_yolo_label sempre retorna uma lista.
    
    Esta propriedade garante que o tipo de retorno é consistente,
    independentemente do conteúdo do arquivo.
    """
    label_file = tmp_path / "test_label.txt"
    label_file.write_text("\n".join(lines))
    
    result = parse_yolo_label(label_file)
    assert isinstance(result, list)
    
    # Todos os elementos da lista devem ser dicionários com as chaves corretas
    for item in result:
        assert isinstance(item, dict)
        assert set(item.keys()) == {'class_id', 'x_center', 'y_center', 'width', 'height'}
