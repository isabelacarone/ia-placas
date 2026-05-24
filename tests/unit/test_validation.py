"""
Testes unitários para o módulo validation.

Valida comportamento da classe YOLOv9Validator incluindo:
- Validação de instalação do YOLOv9 (Requisito 5.1)
- Validação de arquivo DATA_YAML (Requisito 5.9)
- Validação de arquivo de pesos (Requisito 5.5)
- Construção de comando de validação (Requisito 5.1)
- Aplicação de valores padrão (Requisito 5.2)
- Inclusão de flag verbose (Requisito 5.3)
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.validation import YOLOv9Validator


class TestYOLOv9ValidatorInitialization:
    """Testes para inicialização do YOLOv9Validator"""
    
    @patch('src.validation.check_yolov9_installation')
    def test_validator_raises_runtime_error_when_yolov9_not_installed(self, mock_check):
        """
        Requisito 5.4: YOLOv9Validator() deve lançar RuntimeError quando YOLOv9 não instalado
        
        Verifica que a inicialização falha com RuntimeError quando check_yolov9_installation
        retorna False, indicando que o YOLOv9 não está instalado corretamente.
        """
        # Simular YOLOv9 não instalado
        mock_check.return_value = False
        
        # Verificar que RuntimeError é lançado
        with pytest.raises(RuntimeError) as exc_info:
            YOLOv9Validator()
        
        # Verificar que a mensagem contém referência ao diretório YOLOv9
        assert "uvv/yolov9-main/" in str(exc_info.value)
    
    @patch('src.validation.check_yolov9_installation')
    @patch('src.validation.config')
    def test_validator_raises_file_not_found_when_data_yaml_not_exists(self, mock_config, mock_check):
        """
        Requisito 5.9: YOLOv9Validator() deve lançar FileNotFoundError quando DATA_YAML não existe
        
        Verifica que a inicialização falha com FileNotFoundError quando o arquivo
        dados-placas.yaml não existe no caminho esperado.
        """
        # Simular YOLOv9 instalado
        mock_check.return_value = True
        
        # Simular DATA_YAML inexistente
        mock_data_yaml = MagicMock(spec=Path)
        mock_data_yaml.exists.return_value = False
        mock_data_yaml.absolute.return_value = Path("/fake/path/dados-placas.yaml")
        mock_config.DATA_YAML = mock_data_yaml
        
        # Verificar que FileNotFoundError é lançado
        with pytest.raises(FileNotFoundError) as exc_info:
            YOLOv9Validator()
        
        # Verificar que a mensagem contém o caminho completo
        assert "/fake/path/dados-placas.yaml" in str(exc_info.value)
    
    @patch('src.validation.check_yolov9_installation')
    @patch('src.validation.config')
    def test_validator_initializes_successfully_when_setup_valid(self, mock_config, mock_check):
        """
        Verifica que o validador inicializa com sucesso quando o ambiente está configurado
        """
        # Simular YOLOv9 instalado
        mock_check.return_value = True
        
        # Simular DATA_YAML existente
        mock_data_yaml = MagicMock(spec=Path)
        mock_data_yaml.exists.return_value = True
        mock_config.DATA_YAML = mock_data_yaml
        
        # Verificar que inicializa sem exceções
        validator = YOLOv9Validator()
        assert validator is not None
        assert validator.config == mock_config


class TestYOLOv9ValidatorValidate:
    """Testes para o método validate"""
    
    @patch('src.validation.check_yolov9_installation')
    @patch('src.validation.config')
    def test_validate_raises_file_not_found_when_weights_not_exist(self, mock_config, mock_check):
        """
        Requisito 5.5: validate deve lançar FileNotFoundError com caminho completo quando
        arquivo de pesos não existe
        
        Verifica que o método validate falha antes de executar qualquer subprocesso
        quando o arquivo de pesos informado não existe.
        """
        # Configurar mocks para inicialização bem-sucedida
        mock_check.return_value = True
        mock_data_yaml = MagicMock(spec=Path)
        mock_data_yaml.exists.return_value = True
        mock_config.DATA_YAML = mock_data_yaml
        
        # Criar validador
        validator = YOLOv9Validator()
        
        # Tentar validar com arquivo de pesos inexistente
        nonexistent_weights = "/fake/path/to/nonexistent_weights.pt"
        
        with pytest.raises(FileNotFoundError) as exc_info:
            validator.validate(weights=nonexistent_weights)
        
        # Verificar que a mensagem contém o caminho completo
        assert nonexistent_weights in str(exc_info.value)


class TestBuildValidationCommand:
    """Testes para o método _build_validation_command"""
    
    @patch('src.validation.check_yolov9_installation')
    @patch('src.validation.config')
    def test_build_validation_command_contains_all_required_arguments(self, mock_config, mock_check):
        """
        Requisito 5.1: _build_validation_command deve conter todos os argumentos obrigatórios
        
        Verifica que o comando construído contém: --data, --weights, --imgsz, --device,
        --conf-thres, --iou-thres, --name, --batch-size
        """
        # Configurar mocks
        mock_check.return_value = True
        mock_data_yaml = MagicMock(spec=Path)
        mock_data_yaml.exists.return_value = True
        mock_config.DATA_YAML = mock_data_yaml
        mock_config.VAL_SCRIPT = Path("/fake/val.py")
        mock_config.DEFAULT_IMG_SIZE = 640
        mock_config.DEFAULT_VAL_CONF = 0.001
        mock_config.DEFAULT_VAL_IOU = 0.7
        mock_config.DEFAULT_VAL_BATCH = 32
        
        # Criar validador
        validator = YOLOv9Validator()
        
        # Construir comando
        cmd = validator._build_validation_command(
            weights="/fake/weights.pt",
            img_size=640,
            device="0",
            conf_threshold=0.001,
            iou_threshold=0.7,
            name="test_validation",
            batch_size=32,
            verbose=False
        )
        
        # Verificar que todos os argumentos obrigatórios estão presentes
        assert "--data" in cmd
        assert "--weights" in cmd
        assert "--imgsz" in cmd
        assert "--device" in cmd
        assert "--conf-thres" in cmd
        assert "--iou-thres" in cmd
        assert "--name" in cmd
        assert "--batch-size" in cmd
        
        # Verificar valores
        data_idx = cmd.index("--data")
        assert str(mock_data_yaml) == cmd[data_idx + 1]
        
        weights_idx = cmd.index("--weights")
        assert "/fake/weights.pt" == cmd[weights_idx + 1]
        
        imgsz_idx = cmd.index("--imgsz")
        assert "640" == cmd[imgsz_idx + 1]
        
        device_idx = cmd.index("--device")
        assert "0" == cmd[device_idx + 1]
        
        conf_idx = cmd.index("--conf-thres")
        assert "0.001" == cmd[conf_idx + 1]
        
        iou_idx = cmd.index("--iou-thres")
        assert "0.7" == cmd[iou_idx + 1]
        
        name_idx = cmd.index("--name")
        assert "test_validation" == cmd[name_idx + 1]
        
        batch_idx = cmd.index("--batch-size")
        assert "32" == cmd[batch_idx + 1]
    
    @patch('src.validation.check_yolov9_installation')
    @patch('src.validation.config')
    def test_build_validation_command_includes_verbose_when_true(self, mock_config, mock_check):
        """
        Requisito 5.3: --verbose deve ser incluído no comando quando verbose=True
        
        Verifica que a flag --verbose é adicionada ao comando quando o parâmetro
        verbose é True.
        """
        # Configurar mocks
        mock_check.return_value = True
        mock_data_yaml = MagicMock(spec=Path)
        mock_data_yaml.exists.return_value = True
        mock_config.DATA_YAML = mock_data_yaml
        mock_config.VAL_SCRIPT = Path("/fake/val.py")
        
        # Criar validador
        validator = YOLOv9Validator()
        
        # Construir comando com verbose=True
        cmd_verbose = validator._build_validation_command(
            weights="/fake/weights.pt",
            img_size=640,
            device="0",
            conf_threshold=0.001,
            iou_threshold=0.7,
            name="test_validation",
            batch_size=32,
            verbose=True
        )
        
        # Verificar que --verbose está presente
        assert "--verbose" in cmd_verbose
        
        # Construir comando com verbose=False
        cmd_not_verbose = validator._build_validation_command(
            weights="/fake/weights.pt",
            img_size=640,
            device="0",
            conf_threshold=0.001,
            iou_threshold=0.7,
            name="test_validation",
            batch_size=32,
            verbose=False
        )
        
        # Verificar que --verbose não está presente
        assert "--verbose" not in cmd_not_verbose
    
    @patch('src.validation.check_yolov9_installation')
    @patch('src.validation.config')
    def test_build_validation_command_uses_python_executable(self, mock_config, mock_check):
        """
        Verifica que o comando usa sys.executable para invocar o script Python
        """
        # Configurar mocks
        mock_check.return_value = True
        mock_data_yaml = MagicMock(spec=Path)
        mock_data_yaml.exists.return_value = True
        mock_config.DATA_YAML = mock_data_yaml
        mock_config.VAL_SCRIPT = Path("/fake/val.py")
        
        # Criar validador
        validator = YOLOv9Validator()
        
        # Construir comando
        cmd = validator._build_validation_command(
            weights="/fake/weights.pt",
            img_size=640,
            device="0",
            conf_threshold=0.001,
            iou_threshold=0.7,
            name="test_validation",
            batch_size=32,
            verbose=False
        )
        
        # Verificar que o primeiro elemento é sys.executable
        assert cmd[0] == sys.executable
        
        # Verificar que o segundo elemento é o script de validação
        assert cmd[1] == str(mock_config.VAL_SCRIPT)


class TestDefaultValues:
    """Testes para aplicação de valores padrão"""
    
    @patch('src.validation.print_config_summary')
    @patch('src.validation.print_header')
    @patch('src.validation.check_yolov9_installation')
    @patch('src.validation.config')
    @patch('src.validation.subprocess.run')
    def test_validate_applies_default_values(self, mock_subprocess, mock_config, mock_check, mock_print_header, mock_print_summary):
        """
        Requisito 5.2: valores padrão devem ser aplicados quando não especificados
        
        Verifica que img_size=640, conf_threshold=0.001, iou_threshold=0.7, batch_size=32
        são aplicados quando os parâmetros não são fornecidos.
        """
        # Configurar mocks
        mock_check.return_value = True
        mock_data_yaml = MagicMock(spec=Path)
        mock_data_yaml.exists.return_value = True
        mock_config.DATA_YAML = mock_data_yaml
        mock_config.VAL_SCRIPT = Path("/fake/val.py")
        mock_config.YOLOV9_DIR = Path("/fake/yolov9")
        
        # Configurar valores padrão
        mock_config.DEFAULT_IMG_SIZE = 640
        mock_config.DEFAULT_VAL_CONF = 0.001
        mock_config.DEFAULT_VAL_IOU = 0.7
        mock_config.DEFAULT_VAL_BATCH = 32
        
        # Configurar subprocess para retornar sucesso
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        # Criar validador
        validator = YOLOv9Validator()
        
        # Criar arquivo de pesos temporário (mock)
        with patch('src.validation.Path') as mock_path_class:
            mock_weights_path = MagicMock(spec=Path)
            mock_weights_path.exists.return_value = True
            mock_weights_path.absolute.return_value = Path("/fake/weights.pt")
            mock_path_class.return_value = mock_weights_path
            
            # Chamar validate sem especificar parâmetros opcionais
            validator.validate(weights="/fake/weights.pt")
        
        # Verificar que subprocess.run foi chamado
        assert mock_subprocess.called
        
        # Obter o comando passado para subprocess.run
        call_args = mock_subprocess.call_args
        cmd = call_args[0][0]
        
        # Verificar que os valores padrão foram aplicados
        imgsz_idx = cmd.index("--imgsz")
        assert cmd[imgsz_idx + 1] == "640"
        
        conf_idx = cmd.index("--conf-thres")
        assert cmd[conf_idx + 1] == "0.001"
        
        iou_idx = cmd.index("--iou-thres")
        assert cmd[iou_idx + 1] == "0.7"
        
        batch_idx = cmd.index("--batch-size")
        assert cmd[batch_idx + 1] == "32"
    
    @patch('src.validation.check_yolov9_installation')
    @patch('src.validation.config')
    @patch('src.validation.subprocess.run')
    def test_validate_uses_provided_values_over_defaults(self, mock_subprocess, mock_config, mock_check):
        """
        Verifica que valores fornecidos pelo usuário sobrescrevem os padrões
        """
        # Configurar mocks
        mock_check.return_value = True
        mock_data_yaml = MagicMock(spec=Path)
        mock_data_yaml.exists.return_value = True
        mock_config.DATA_YAML = mock_data_yaml
        mock_config.VAL_SCRIPT = Path("/fake/val.py")
        mock_config.YOLOV9_DIR = Path("/fake/yolov9")
        
        # Configurar valores padrão
        mock_config.DEFAULT_IMG_SIZE = 640
        mock_config.DEFAULT_VAL_CONF = 0.001
        mock_config.DEFAULT_VAL_IOU = 0.7
        mock_config.DEFAULT_VAL_BATCH = 32
        
        # Configurar subprocess para retornar sucesso
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        # Criar validador
        validator = YOLOv9Validator()
        
        # Criar arquivo de pesos temporário (mock)
        with patch('src.validation.Path') as mock_path_class:
            mock_weights_path = MagicMock(spec=Path)
            mock_weights_path.exists.return_value = True
            mock_weights_path.absolute.return_value = Path("/fake/weights.pt")
            mock_path_class.return_value = mock_weights_path
            
            # Chamar validate com valores customizados
            validator.validate(
                weights="/fake/weights.pt",
                img_size=1280,
                conf_threshold=0.5,
                iou_threshold=0.6,
                batch_size=16
            )
        
        # Verificar que subprocess.run foi chamado
        assert mock_subprocess.called
        
        # Obter o comando passado para subprocess.run
        call_args = mock_subprocess.call_args
        cmd = call_args[0][0]
        
        # Verificar que os valores customizados foram usados
        imgsz_idx = cmd.index("--imgsz")
        assert cmd[imgsz_idx + 1] == "1280"
        
        conf_idx = cmd.index("--conf-thres")
        assert cmd[conf_idx + 1] == "0.5"
        
        iou_idx = cmd.index("--iou-thres")
        assert cmd[iou_idx + 1] == "0.6"
        
        batch_idx = cmd.index("--batch-size")
        assert cmd[batch_idx + 1] == "16"
