"""
Testes de propriedade para validação de formato de placas usando Hypothesis.

Feature: detector-placas-yolov9
Properties:
- Property 7: classificação de Placa_Antiga é correta para todos os inputs válidos
- Property 8: classificação de Placa_Mercosul é correta para todos os inputs válidos
- Property 9: strings com comprimento diferente de 7 são sempre inválidas

Valida: Requisitos 9.1, 9.2, 9.5
"""

import pytest
import string
from hypothesis import given, settings, strategies as st
from src.detection import PlateCharacterProcessor


# Definir alfabetos conforme especificação
LETTERS = string.ascii_uppercase.replace('O', '')  # A-Z exceto O
DIGITS = string.digits  # 0-9


# Feature: detector-placas-yolov9, Property 7: classificação de Placa_Antiga
@given(
    st.text(alphabet=LETTERS, min_size=3, max_size=3),
    st.text(alphabet=DIGITS, min_size=4, max_size=4)
)
@settings(max_examples=200)
def test_placa_antiga_classification_is_correct(letters, digits):
    """
    Property 7: Para qualquer string de 3 letras maiúsculas + 4 dígitos,
    _validate_plate_format retorna valid_format=True e plate_type='Placa_Antiga'.
    
    Esta propriedade garante que todas as placas no formato antigo brasileiro
    (ABC1234) são corretamente identificadas, independentemente dos caracteres
    específicos usados.
    """
    processor = PlateCharacterProcessor()
    plate = letters + digits
    
    result = processor._validate_plate_format(plate)
    
    assert result['valid_format'] is True, \
        f"Placa '{plate}' deveria ser válida mas foi marcada como inválida"
    assert result['plate_type'] == 'Placa_Antiga', \
        f"Placa '{plate}' deveria ser Placa_Antiga mas foi classificada como {result['plate_type']}"


@given(st.text(alphabet=LETTERS + DIGITS, min_size=7, max_size=7))
@settings(max_examples=200)
def test_placa_antiga_pattern_recognition(plate_text):
    """
    Property 7a: Apenas placas com padrão exato [A-Z]{3}[0-9]{4} são Placa_Antiga.
    
    Esta propriedade garante que a validação é precisa e não aceita
    variações do formato.
    """
    processor = PlateCharacterProcessor()
    
    # Verificar se o texto corresponde ao padrão Placa_Antiga
    is_placa_antiga_pattern = (
        len(plate_text) == 7 and
        all(c in LETTERS for c in plate_text[:3]) and
        all(c in DIGITS for c in plate_text[3:])
    )
    
    result = processor._validate_plate_format(plate_text)
    
    if is_placa_antiga_pattern:
        assert result['valid_format'] is True
        assert result['plate_type'] == 'Placa_Antiga'
    else:
        # Se não é Placa_Antiga, pode ser Placa_Mercosul ou inválida
        if result['plate_type'] == 'Placa_Antiga':
            pytest.fail(f"Placa '{plate_text}' não deveria ser classificada como Placa_Antiga")


# Feature: detector-placas-yolov9, Property 8: classificação de Placa_Mercosul
@given(
    st.text(alphabet=LETTERS, min_size=3, max_size=3),
    st.text(alphabet=DIGITS, min_size=1, max_size=1),
    st.text(alphabet=LETTERS, min_size=1, max_size=1),
    st.text(alphabet=DIGITS, min_size=2, max_size=2)
)
@settings(max_examples=200)
def test_placa_mercosul_classification_is_correct(letters1, digit1, letter, digits2):
    """
    Property 8: Para qualquer string de 3 letras + 1 dígito + 1 letra + 2 dígitos,
    retorna valid_format=True e plate_type='Placa_Mercosul'.
    
    Esta propriedade garante que todas as placas no formato Mercosul
    (ABC1D23) são corretamente identificadas.
    """
    processor = PlateCharacterProcessor()
    plate = letters1 + digit1 + letter + digits2
    
    result = processor._validate_plate_format(plate)
    
    assert result['valid_format'] is True, \
        f"Placa '{plate}' deveria ser válida mas foi marcada como inválida"
    assert result['plate_type'] == 'Placa_Mercosul', \
        f"Placa '{plate}' deveria ser Placa_Mercosul mas foi classificada como {result['plate_type']}"


@given(st.text(alphabet=LETTERS + DIGITS, min_size=7, max_size=7))
@settings(max_examples=200)
def test_placa_mercosul_pattern_recognition(plate_text):
    """
    Property 8a: Apenas placas com padrão exato [A-Z]{3}[0-9][A-Z][0-9]{2} são Placa_Mercosul.
    
    Esta propriedade garante que a validação é precisa e não aceita
    variações do formato.
    """
    processor = PlateCharacterProcessor()
    
    # Verificar se o texto corresponde ao padrão Placa_Mercosul
    is_placa_mercosul_pattern = (
        len(plate_text) == 7 and
        all(c in LETTERS for c in plate_text[:3]) and
        plate_text[3] in DIGITS and
        plate_text[4] in LETTERS and
        all(c in DIGITS for c in plate_text[5:])
    )
    
    result = processor._validate_plate_format(plate_text)
    
    if is_placa_mercosul_pattern:
        assert result['valid_format'] is True
        assert result['plate_type'] == 'Placa_Mercosul'
    else:
        # Se não é Placa_Mercosul, pode ser Placa_Antiga ou inválida
        if result['plate_type'] == 'Placa_Mercosul':
            pytest.fail(f"Placa '{plate_text}' não deveria ser classificada como Placa_Mercosul")


# Feature: detector-placas-yolov9, Property 9: strings com comprimento diferente de 7 são inválidas
@given(st.text(min_size=0, max_size=20).filter(lambda s: len(s) != 7))
@settings(max_examples=200)
def test_wrong_length_always_invalid(text):
    """
    Property 9: Para qualquer string com comprimento diferente de 7,
    retorna valid_format=False e plate_type=None.
    
    Esta propriedade garante que a validação rejeita imediatamente
    textos com comprimento incorreto, sem tentar validar o padrão.
    """
    processor = PlateCharacterProcessor()
    
    result = processor._validate_plate_format(text)
    
    assert result['valid_format'] is False, \
        f"Texto '{text}' com comprimento {len(text)} deveria ser inválido"
    assert result['plate_type'] is None, \
        f"Texto '{text}' com comprimento {len(text)} não deveria ter plate_type"


@given(st.integers(min_value=0, max_value=20).filter(lambda n: n != 7))
@settings(max_examples=200)
def test_wrong_length_with_valid_characters(length):
    """
    Property 9a: Mesmo com caracteres válidos, comprimento != 7 é inválido.
    
    Esta propriedade garante que o comprimento é verificado antes
    do padrão de caracteres.
    """
    processor = PlateCharacterProcessor()
    
    # Criar string com caracteres válidos mas comprimento errado
    if length <= 3:
        text = LETTERS[:length] if length > 0 else ""
    elif length <= 7:
        text = LETTERS[:3] + DIGITS[:length-3]
    else:
        text = LETTERS[:3] + DIGITS[:4] + LETTERS[:length-7]
    
    result = processor._validate_plate_format(text)
    
    assert result['valid_format'] is False
    assert result['plate_type'] is None


# Testes de casos extremos
def test_empty_string_is_invalid():
    """
    Caso extremo: String vazia é inválida.
    """
    processor = PlateCharacterProcessor()
    result = processor._validate_plate_format("")
    
    assert result['valid_format'] is False
    assert result['plate_type'] is None


@given(st.text(alphabet=string.ascii_lowercase, min_size=7, max_size=7))
@settings(max_examples=200)
def test_lowercase_letters_are_invalid(text):
    """
    Caso extremo: Letras minúsculas não são aceitas.
    
    Esta propriedade garante que apenas letras maiúsculas são válidas,
    conforme especificação do formato de placas brasileiras.
    """
    processor = PlateCharacterProcessor()
    result = processor._validate_plate_format(text)
    
    # Letras minúsculas não correspondem ao padrão, então devem ser inválidas
    # (a menos que por acaso formem um padrão válido, o que é improvável)
    # Como usamos apenas minúsculas, nunca será válido
    assert result['valid_format'] is False
    assert result['plate_type'] is None


@given(st.text(alphabet=string.punctuation + ' ', min_size=7, max_size=7))
@settings(max_examples=200)
def test_special_characters_are_invalid(text):
    """
    Caso extremo: Caracteres especiais e espaços não são aceitos.
    """
    processor = PlateCharacterProcessor()
    result = processor._validate_plate_format(text)
    
    assert result['valid_format'] is False
    assert result['plate_type'] is None


# Teste de invariante: _validate_plate_format sempre retorna dict com chaves corretas
@given(st.text(min_size=0, max_size=20))
@settings(max_examples=200)
def test_validate_plate_format_always_returns_correct_structure(text):
    """
    Invariante: _validate_plate_format sempre retorna dict com chaves
    'valid_format' (bool) e 'plate_type' (str ou None).
    
    Esta propriedade garante que a estrutura de retorno é consistente,
    independentemente da entrada.
    """
    processor = PlateCharacterProcessor()
    result = processor._validate_plate_format(text)
    
    assert isinstance(result, dict)
    assert set(result.keys()) == {'valid_format', 'plate_type'}
    assert isinstance(result['valid_format'], bool)
    assert result['plate_type'] in ['Placa_Antiga', 'Placa_Mercosul', None]


# Teste de exclusividade mútua: uma placa não pode ser ambos os tipos
@given(st.text(alphabet=LETTERS + DIGITS, min_size=7, max_size=7))
@settings(max_examples=200)
def test_plate_cannot_be_both_types(text):
    """
    Invariante: Uma placa não pode ser simultaneamente Placa_Antiga e Placa_Mercosul.
    
    Esta propriedade garante que a classificação é mutuamente exclusiva.
    """
    processor = PlateCharacterProcessor()
    result = processor._validate_plate_format(text)
    
    # Se é válida, deve ser exatamente um tipo
    if result['valid_format']:
        assert result['plate_type'] in ['Placa_Antiga', 'Placa_Mercosul']
    else:
        assert result['plate_type'] is None


# Teste de determinismo
@given(st.text(min_size=0, max_size=20))
@settings(max_examples=200)
def test_validate_plate_format_is_deterministic(text):
    """
    Invariante: _validate_plate_format é determinístico.
    
    Esta propriedade garante que múltiplas chamadas com a mesma entrada
    produzem o mesmo resultado.
    """
    processor = PlateCharacterProcessor()
    
    result1 = processor._validate_plate_format(text)
    result2 = processor._validate_plate_format(text)
    
    assert result1 == result2
