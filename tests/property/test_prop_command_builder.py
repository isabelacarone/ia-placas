"""
Testes de propriedade para construção de comandos usando Hypothesis.

Feature: detector-placas-yolov9
Properties:
- Property 10: construção de comando de treinamento contém todos os argumentos obrigatórios
- Property 11: resume é incluído no comando somente quando fornecido

Valida: Requisitos 4.1, 4.3
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from hypothesis import given, settings, strategies as st, assume
from src.training import YOLOv9Trainer


# Estratégias para gerar parâmetros válidos de treinamento
valid_device_strategy = st.one_of(
    st.just('cpu'),
    st.integers(min_value=0, max_value=9).map(str),
    st.integers(min_value=0, max_value=9).map(lambda n: f"cuda:{n}")
)

epochs_strategy = st.integers(min_value=1, max_value=1000)
batch_size_strategy = st.integers(min_value=1, max_value=128)
img_size_strategy = st.sampled_from([320, 416, 512, 640, 1280])
workers_strategy = st.integers(min_value=0, max_value=16)
name_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='_-'),
    min_size=1,
    max_size=50
).filter(lambda s: s and not s.startswith('-'))


# Feature: detector-placas-yolov9, Property 10: comando contém todos os argumentos obrigatórios
@given(
    epochs=epochs_strategy,
    batch_size=batch_size_strategy,
    img_size=img_size_strategy,
    device=valid_device_strategy,
    workers=workers_strategy,
    name=name_strategy
)
@settings(max_examples=100)
@patch('src.training.check_yolov9_installation')
@patch('src.training.config')
def test_build_train_command_contains_all_mandatory_arguments(
    mock_config, mock_check,
    epochs, batch_size, img_size, device, workers, name
):
    """
    Property 10: Para qualquer combinação válida de parâmetros de treinamento,
    _build_train_command retorna lista contendo todos os 11 argumentos obrigatórios.
    
    Argumentos obrigatórios:
    --workers, --device, --batch-size, --data, --img, --cfg, --weights,
    --name, --hyp, --epochs, --noplots
    
    Esta propriedade garante que o comando construído sempre inclui todos
    os argumentos necessários para o treinamento, independentemente dos
    valores específicos fornecidos.
    """
    # Configurar mocks
    mock_check.return_value = True
    mock_data_yaml = MagicMock(spec=Path)
    mock_data_yaml.exists.return_value = True
    mock_config.DATA_YAML = mock_data_yaml
    mock_config.TRAIN_SCRIPT = Path("/fake/train_dual.py")
    mock_config.MODEL_YAML = Path("/fake/model.yaml")
    mock_config.HYPERPARAMS_YAML = Path("/fake/hyp.yaml")
    
    # Criar trainer
    trainer = YOLOv9Trainer()
    
    # Construir comando
    cmd = trainer._build_train_command(
        epochs=epochs,
        batch_size=batch_size,
        img_size=img_size,
        device=device,
        workers=workers,
        name=name,
        resume=None,
        weights=""
    )
    
    # Verificar que todos os 11 argumentos obrigatórios estão presentes
    mandatory_args = [
        '--workers',
        '--device',
        '--batch-size',
        '--data',
        '--img',
        '--cfg',
        '--weights',
        '--name',
        '--hyp',
        '--epochs',
        '--noplots'
    ]
    
    for arg in mandatory_args:
        assert arg in cmd, f"Argumento obrigatório '{arg}' não encontrado no comando"
    
    # Verificar que os valores foram incluídos corretamente
    assert str(epochs) in cmd
    assert str(batch_size) in cmd
    assert str(img_size) in cmd
    assert name in cmd
    
    # Verificar que o comando começa com sys.executable e o script
    assert cmd[0] == sys.executable
    assert cmd[1] == str(mock_config.TRAIN_SCRIPT)


@given(
    epochs=epochs_strategy,
    batch_size=batch_size_strategy,
    img_size=img_size_strategy,
    device=valid_device_strategy,
    workers=workers_strategy,
    name=name_strategy
)
@settings(max_examples=100)
@patch('src.training.check_yolov9_installation')
@patch('src.training.config')
def test_build_train_command_values_are_correct(
    mock_config, mock_check,
    epochs, batch_size, img_size, device, workers, name
):
    """
    Property 10a: Os valores dos argumentos correspondem aos parâmetros fornecidos.
    
    Esta propriedade garante que os valores não são apenas incluídos,
    mas estão corretos e na posição adequada.
    """
    # Configurar mocks
    mock_check.return_value = True
    mock_data_yaml = MagicMock(spec=Path)
    mock_data_yaml.exists.return_value = True
    mock_config.DATA_YAML = mock_data_yaml
    mock_config.TRAIN_SCRIPT = Path("/fake/train_dual.py")
    mock_config.MODEL_YAML = Path("/fake/model.yaml")
    mock_config.HYPERPARAMS_YAML = Path("/fake/hyp.yaml")
    
    # Criar trainer
    trainer = YOLOv9Trainer()
    
    # Construir comando
    cmd = trainer._build_train_command(
        epochs=epochs,
        batch_size=batch_size,
        img_size=img_size,
        device=device,
        workers=workers,
        name=name,
        resume=None,
        weights=""
    )
    
    # Verificar valores específicos
    epochs_idx = cmd.index('--epochs')
    assert cmd[epochs_idx + 1] == str(epochs)
    
    batch_idx = cmd.index('--batch-size')
    assert cmd[batch_idx + 1] == str(batch_size)
    
    img_idx = cmd.index('--img')
    assert cmd[img_idx + 1] == str(img_size)
    
    workers_idx = cmd.index('--workers')
    assert cmd[workers_idx + 1] == str(workers)
    
    name_idx = cmd.index('--name')
    assert cmd[name_idx + 1] == name


# Feature: detector-placas-yolov9, Property 11: resume é incluído somente quando fornecido
@given(
    resume_path=st.one_of(
        st.none(),
        st.text(min_size=1, max_size=100).map(lambda s: f"/fake/path/{s}.pt")
    )
)
@settings(max_examples=100)
@patch('src.training.check_yolov9_installation')
@patch('src.training.config')
def test_build_train_command_resume_inclusion(mock_config, mock_check, resume_path):
    """
    Property 11: Quando resume é não-None, --resume está presente na lista;
    quando resume é None, --resume não aparece.
    
    Esta propriedade garante que o argumento --resume é incluído condicionalmente,
    apenas quando o usuário deseja retomar um treinamento anterior.
    """
    # Configurar mocks
    mock_check.return_value = True
    mock_data_yaml = MagicMock(spec=Path)
    mock_data_yaml.exists.return_value = True
    mock_config.DATA_YAML = mock_data_yaml
    mock_config.TRAIN_SCRIPT = Path("/fake/train_dual.py")
    mock_config.MODEL_YAML = Path("/fake/model.yaml")
    mock_config.HYPERPARAMS_YAML = Path("/fake/hyp.yaml")
    
    # Criar trainer
    trainer = YOLOv9Trainer()
    
    # Construir comando
    cmd = trainer._build_train_command(
        epochs=100,
        batch_size=8,
        img_size=640,
        device='0',
        workers=0,
        name='test',
        resume=resume_path,
        weights=""
    )
    
    # Verificar presença de --resume
    if resume_path is not None:
        assert '--resume' in cmd, \
            f"--resume deveria estar presente quando resume='{resume_path}'"
        
        # Verificar que o valor está correto
        resume_idx = cmd.index('--resume')
        assert cmd[resume_idx + 1] == resume_path
    else:
        assert '--resume' not in cmd, \
            "--resume não deveria estar presente quando resume=None"


@given(
    resume_path=st.text(min_size=1, max_size=100).map(lambda s: f"/fake/path/{s}.pt")
)
@settings(max_examples=100)
@patch('src.training.check_yolov9_installation')
@patch('src.training.config')
def test_build_train_command_resume_value_is_correct(mock_config, mock_check, resume_path):
    """
    Property 11a: Quando --resume está presente, seu valor corresponde ao caminho fornecido.
    
    Esta propriedade garante que o caminho do checkpoint é passado corretamente
    para o script de treinamento.
    """
    # Configurar mocks
    mock_check.return_value = True
    mock_data_yaml = MagicMock(spec=Path)
    mock_data_yaml.exists.return_value = True
    mock_config.DATA_YAML = mock_data_yaml
    mock_config.TRAIN_SCRIPT = Path("/fake/train_dual.py")
    mock_config.MODEL_YAML = Path("/fake/model.yaml")
    mock_config.HYPERPARAMS_YAML = Path("/fake/hyp.yaml")
    
    # Criar trainer
    trainer = YOLOv9Trainer()
    
    # Construir comando com resume
    cmd = trainer._build_train_command(
        epochs=100,
        batch_size=8,
        img_size=640,
        device='0',
        workers=0,
        name='test',
        resume=resume_path,
        weights=""
    )
    
    # Verificar que --resume está presente e com valor correto
    assert '--resume' in cmd
    resume_idx = cmd.index('--resume')
    assert cmd[resume_idx + 1] == resume_path


@patch('src.training.check_yolov9_installation')
@patch('src.training.config')
def test_build_train_command_resume_none_excludes_argument(mock_config, mock_check):
    """
    Property 11b: Quando resume=None, --resume não aparece em nenhuma posição do comando.
    
    Esta propriedade garante que não há inclusão acidental do argumento
    quando não é necessário.
    """
    # Configurar mocks
    mock_check.return_value = True
    mock_data_yaml = MagicMock(spec=Path)
    mock_data_yaml.exists.return_value = True
    mock_config.DATA_YAML = mock_data_yaml
    mock_config.TRAIN_SCRIPT = Path("/fake/train_dual.py")
    mock_config.MODEL_YAML = Path("/fake/model.yaml")
    mock_config.HYPERPARAMS_YAML = Path("/fake/hyp.yaml")
    
    # Criar trainer
    trainer = YOLOv9Trainer()
    
    # Construir comando sem resume
    cmd = trainer._build_train_command(
        epochs=100,
        batch_size=8,
        img_size=640,
        device='0',
        workers=0,
        name='test',
        resume=None,
        weights=""
    )
    
    # Verificar que --resume não está em nenhuma posição
    assert '--resume' not in cmd
    assert 'None' not in cmd  # Garantir que None não foi convertido para string


# Teste de invariante: comando sempre começa com executável Python e script
@given(
    epochs=epochs_strategy,
    batch_size=batch_size_strategy,
    img_size=img_size_strategy,
    device=valid_device_strategy,
    workers=workers_strategy,
    name=name_strategy,
    resume=st.one_of(st.none(), st.text(min_size=1, max_size=50))
)
@settings(max_examples=100)
@patch('src.training.check_yolov9_installation')
@patch('src.training.config')
def test_build_train_command_structure_is_consistent(
    mock_config, mock_check,
    epochs, batch_size, img_size, device, workers, name, resume
):
    """
    Invariante: O comando sempre começa com sys.executable seguido do script de treinamento.
    
    Esta propriedade garante que a estrutura básica do comando é consistente,
    permitindo execução correta via subprocess.
    """
    # Configurar mocks
    mock_check.return_value = True
    mock_data_yaml = MagicMock(spec=Path)
    mock_data_yaml.exists.return_value = True
    mock_config.DATA_YAML = mock_data_yaml
    mock_config.TRAIN_SCRIPT = Path("/fake/train_dual.py")
    mock_config.MODEL_YAML = Path("/fake/model.yaml")
    mock_config.HYPERPARAMS_YAML = Path("/fake/hyp.yaml")
    
    # Criar trainer
    trainer = YOLOv9Trainer()
    
    # Construir comando
    cmd = trainer._build_train_command(
        epochs=epochs,
        batch_size=batch_size,
        img_size=img_size,
        device=device,
        workers=workers,
        name=name,
        resume=resume,
        weights=""
    )
    
    # Verificar estrutura
    assert isinstance(cmd, list)
    assert len(cmd) >= 2
    assert cmd[0] == sys.executable
    assert cmd[1] == str(mock_config.TRAIN_SCRIPT)


# Teste de determinismo
@given(
    epochs=epochs_strategy,
    batch_size=batch_size_strategy,
    img_size=img_size_strategy,
    device=valid_device_strategy,
    workers=workers_strategy,
    name=name_strategy,
    resume=st.one_of(st.none(), st.text(min_size=1, max_size=50))
)
@settings(max_examples=100)
@patch('src.training.check_yolov9_installation')
@patch('src.training.config')
def test_build_train_command_is_deterministic(
    mock_config, mock_check,
    epochs, batch_size, img_size, device, workers, name, resume
):
    """
    Invariante: _build_train_command é determinístico.
    
    Esta propriedade garante que múltiplas chamadas com os mesmos parâmetros
    produzem o mesmo comando.
    """
    # Configurar mocks
    mock_check.return_value = True
    mock_data_yaml = MagicMock(spec=Path)
    mock_data_yaml.exists.return_value = True
    mock_config.DATA_YAML = mock_data_yaml
    mock_config.TRAIN_SCRIPT = Path("/fake/train_dual.py")
    mock_config.MODEL_YAML = Path("/fake/model.yaml")
    mock_config.HYPERPARAMS_YAML = Path("/fake/hyp.yaml")
    
    # Criar trainer
    trainer = YOLOv9Trainer()
    
    # Construir comando duas vezes
    cmd1 = trainer._build_train_command(
        epochs=epochs,
        batch_size=batch_size,
        img_size=img_size,
        device=device,
        workers=workers,
        name=name,
        resume=resume,
        weights=""
    )
    
    cmd2 = trainer._build_train_command(
        epochs=epochs,
        batch_size=batch_size,
        img_size=img_size,
        device=device,
        workers=workers,
        name=name,
        resume=resume,
        weights=""
    )
    
    assert cmd1 == cmd2
