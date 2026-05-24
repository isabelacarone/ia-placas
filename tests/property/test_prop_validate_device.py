"""
Testes de propriedade para validate_device usando Hypothesis.

Feature: detector-placas-yolov9
Properties:
- Property 1: validate_device normaliza corretamente entradas válidas
- Property 2: validate_device rejeita entradas inválidas

Valida: Requisitos 2.2, 2.3, 10.3
"""

import pytest
from hypothesis import given, settings, strategies as st
from src.utils import validate_device


# Feature: detector-placas-yolov9, Property 1: validate_device normaliza entradas válidas
@given(st.integers(min_value=0, max_value=9).map(str))
@settings(max_examples=100)
def test_validate_device_digit_returns_itself(digit_str):
    """
    Property 1a: Para qualquer dígito '0'–'9', validate_device retorna o próprio dígito.
    
    Esta propriedade garante que strings numéricas de um único dígito são aceitas
    e retornadas sem modificação, representando índices de GPU válidos.
    """
    result = validate_device(digit_str)
    assert result == digit_str
    assert isinstance(result, str)


@given(st.integers(min_value=0, max_value=9).map(lambda n: f"cuda:{n}"))
@settings(max_examples=100)
def test_validate_device_cuda_format_extracts_digit(cuda_str):
    """
    Property 1b: Para qualquer 'cuda:N' onde N é dígito, retorna apenas 'N'.
    
    Esta propriedade garante que o formato CUDA completo é normalizado para
    apenas o índice numérico da GPU, mantendo consistência na representação.
    """
    expected = cuda_str.split(":")[1]
    result = validate_device(cuda_str)
    assert result == expected
    assert isinstance(result, str)
    assert result.isdigit()


def test_validate_device_cpu_always_returns_cpu():
    """
    Property 1c: validate_device('cpu') sempre retorna 'cpu'.
    
    Esta propriedade garante que a string 'cpu' é sempre aceita e retornada
    sem modificação, representando execução em CPU.
    """
    result = validate_device('cpu')
    assert result == 'cpu'
    assert isinstance(result, str)


# Feature: detector-placas-yolov9, Property 2: validate_device rejeita entradas inválidas
@given(
    st.text(
        alphabet=st.characters(blacklist_categories=('Cs', 'Cc')),  # Excluir caracteres de controle
        min_size=1,
        max_size=20
    ).filter(
        lambda s: (
            s.strip() != 'cpu' and
            not (len(s.strip()) == 1 and s.strip().isdigit()) and
            not (s.strip().startswith('cuda:') and len(s.strip()) == 6 and s.strip()[5].isdigit()) and
            len(s.strip()) > 0  # Evitar strings apenas com espaços
        )
    )
)
@settings(max_examples=100)
def test_validate_device_invalid_raises_value_error(invalid_str):
    """
    Property 2: Para qualquer string que não seja 'cpu', dígito isolado ou 'cuda:N',
    lança ValueError com o valor inválido na mensagem.
    
    Esta propriedade garante que entradas inválidas são rejeitadas de forma consistente,
    fornecendo feedback claro ao usuário sobre o valor problemático.
    
    Nota: validate_device faz strip() na entrada, então testamos considerando isso.
    """
    with pytest.raises(ValueError) as exc_info:
        validate_device(invalid_str)
    
    # A mensagem de erro contém o valor após strip e lower
    stripped_value = invalid_str.strip().lower()
    error_message = str(exc_info.value)
    assert stripped_value in error_message


def test_validate_device_rejects_common_invalid_inputs():
    """
    Property 2 (casos específicos): Testa rejeição de entradas inválidas comuns.
    
    Complementa o teste baseado em propriedade com casos específicos que são
    frequentemente tentados por usuários.
    """
    invalid_inputs = [
        'gpu',           # Termo comum mas inválido
        'cuda',          # Prefixo sem índice
        'cuda:',         # Prefixo incompleto
        'abc',           # String arbitrária
        '10',            # Dígito fora do intervalo 0-9
        'cuda:10',       # CUDA com índice de dois dígitos
        'CPU',           # Case incorreto (mas será aceito após lower())
        '0gpu',          # Formato misto
        'cuda:a',        # CUDA com letra
    ]
    
    for invalid_input in invalid_inputs:
        # CPU em maiúsculas é aceito após lower(), então pulamos
        if invalid_input.lower().strip() == 'cpu':
            continue
            
        with pytest.raises(ValueError) as exc_info:
            validate_device(invalid_input)
        # Verificar que a mensagem contém o valor após strip/lower
        assert invalid_input.lower().strip() in str(exc_info.value).lower()


# Teste de invariante: validate_device sempre retorna string
@given(
    st.one_of(
        st.just('cpu'),
        st.integers(min_value=0, max_value=9).map(str),
        st.integers(min_value=0, max_value=9).map(lambda n: f"cuda:{n}")
    )
)
@settings(max_examples=100)
def test_validate_device_always_returns_string(valid_device):
    """
    Invariante: validate_device sempre retorna uma string quando bem-sucedido.
    
    Esta propriedade garante que o tipo de retorno é consistente para todas
    as entradas válidas.
    """
    result = validate_device(valid_device)
    assert isinstance(result, str)
    assert len(result) > 0
