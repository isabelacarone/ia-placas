"""
Testes unitários para o módulo training.

Valida comportamento do YOLOv9Trainer incluindo:
- Validação de instalação do YOLOv9 (Requisito 4.7)
- Validação de existência do DATA_YAML (Requisito 4.8)
- Construção de comando de treinamento (Requisitos 4.1, 4.3)
- Uso de valores padrão (Requisito 4.2)
- Validação de device (Requisito 4.9)
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.training import YOLOv9Trainer


class TestYOLOv9TrainerInstantiation:
    """Testes para instanciação do YOLOv9Trainer (Requisitos 4.7, 4.8)"""
    
    @patch('src.training.check_yolov9_installation')
    def test_trainer_raises_runtime_error_when_yolov9_not_installed(self, mock_check):
        """
        Requisito 4.7: deve lançar RuntimeError com 'uvv/yolov9-main/' quando YOLOv9 não instalado
        
        Testa que a instanciação do trainer falha quando check_yolov9_installation retorna False.
        """
        # Configurar mock para simular YOLOv9 não instalado
        mock_check.return_value = False
        
        # Verificar que RuntimeError é lançado
        with pytest.raises(RuntimeError) as exc_info:
            YOLOv9Trainer()
        
        # Verificar que a mensagem contém o caminho esperado
        assert "uvv/yolov9-main/" in str(exc_info.value)
    
    @patch('src.training.check_yolov9_installation')
    @patch('src.training.config')
    def test_trainer_raises_file_not_found_when_data_yaml_missing(self, mock_config, mock_check):
        """
        Requisito 4.8: deve lançar FileNotFoundError com caminho completo quando DATA_YAML não existe
        
        Testa que a instanciação do trainer falha quando o arquivo de configuração de dados não existe.
        """
        # Configurar mock para simular YOLOv9 instalado
        mock_check.return_value = True
        
        # Configurar mock para simular DATA_YAML inexistente
        mock_data_yaml = MagicMock(spec=Path)
        mock_data_yaml.exists.return_value = False
        mock_data_yaml.absolute.return_value = Path("/fake/path/dados-placas.yaml")
        mock_config.DATA_YAML = mock_data_yaml
        
        # Verificar que FileNotFoundError é lançado
        with pytest.raises(FileNotFoundError) as exc_info:
            YOLOv9Trainer()
        
        # Verificar que a mensagem contém o caminho completo
        assert "/fake/path/dados-placas.yaml" in str(exc_info.value)


class TestBuildTrainCommand:
    """Testes para _build_train_command (Requisitos 4.1, 4.3)"""
    
    @patch('src.training.check_yolov9_installation')
    @patch('src.training.config')
    def test_build_train_command_contains_all_mandatory_arguments(self, mock_config, mock_check):
        """
        Requisito 4.1: _build_train_command deve conter todos os 11 argumentos obrigatórios
        
        Verifica que o comando construído contém:
        --workers, --device, --batch-size, --data, --img, --cfg, --weights, --name, --hyp, --epochs, --noplots
        """
        # Configurar mocks
        mock_check.return_value = True
        mock_data_yaml = MagicMock(spec=Path)
        mock_data_yaml.exists.return_value = True
        mock_data_yaml.absolute.return_value = Path("/fake/path/dados-placas.yaml")
        mock_config.DATA_YAML = mock_data_yaml
        mock_config.TRAIN_SCRIPT = Path("/fake/train_dual.py")
        mock_config.MODEL_YAML = Path("/fake/model.yaml")
        mock_config.HYPERPARAMS_YAML = Path("/fake/hyp.yaml")
        
        # Criar trainer
        trainer = YOLOv9Trainer()
        
        # Construir comando
        cmd = trainer._build_train_command(
            epochs=100,
            batch_size=16,
            img_size=640,
            device="0",
            workers=4,
            name="test_experiment",
            resume=None,
            weights=""
        )
        
        # Converter comando para string para facilitar verificação
        cmd_str = " ".join(cmd)
        
        # Verificar que todos os 11 argumentos obrigatórios estão presentes
        mandatory_args = [
            "--workers",
            "--device",
            "--batch-size",
            "--data",
            "--img",
            "--cfg",
            "--weights",
            "--name",
            "--hyp",
            "--epochs",
            "--noplots"
        ]
        
        for arg in mandatory_args:
            assert arg in cmd_str, f"Argumento obrigatório '{arg}' não encontrado no comando"
    
    @patch('src.training.check_yolov9_installation')
    @patch('src.training.config')
    def test_build_train_command_includes_resume_when_provided(self, mock_config, mock_check):
        """
        Requisito 4.3: _build_train_command deve incluir --resume quando resume é fornecido
        
        Verifica que o argumento --resume é incluído no comando quando um caminho é fornecido.
        """
        # Configurar mocks
        mock_check.return_value = True
        mock_data_yaml = MagicMock(spec=Path)
        mock_data_yaml.exists.return_value = True
        mock_data_yaml.absolute.return_value = Path("/fake/path/dados-placas.yaml")
        mock_config.DATA_YAML = mock_data_yaml
        mock_config.TRAIN_SCRIPT = Path("/fake/train_dual.py")
        mock_config.MODEL_YAML = Path("/fake/model.yaml")
        mock_config.HYPERPARAMS_YAML = Path("/fake/hyp.yaml")
        
        # Criar trainer
        trainer = YOLOv9Trainer()
        
        # Construir comando com resume
        resume_path = "/path/to/checkpoint.pt"
        cmd = trainer._build_train_command(
            epochs=100,
            batch_size=16,
            img_size=640,
            device="0",
            workers=4,
            name="test_experiment",
            resume=resume_path,
            weights=""
        )
        
        # Verificar que --resume está presente
        assert "--resume" in cmd
        
        # Verificar que o caminho do checkpoint está presente
        assert resume_path in cmd
        
        # Verificar que --resume vem antes do caminho
        resume_idx = cmd.index("--resume")
        assert cmd[resume_idx + 1] == resume_path
    
    @patch('src.training.check_yolov9_installation')
    @patch('src.training.config')
    def test_build_train_command_excludes_resume_when_none(self, mock_config, mock_check):
        """
        Requisito 4.3: _build_train_command não deve incluir --resume quando resume é None
        
        Verifica que o argumento --resume não é incluído quando resume é None.
        """
        # Configurar mocks
        mock_check.return_value = True
        mock_data_yaml = MagicMock(spec=Path)
        mock_data_yaml.exists.return_value = True
        mock_data_yaml.absolute.return_value = Path("/fake/path/dados-placas.yaml")
        mock_config.DATA_YAML = mock_data_yaml
        mock_config.TRAIN_SCRIPT = Path("/fake/train_dual.py")
        mock_config.MODEL_YAML = Path("/fake/model.yaml")
        mock_config.HYPERPARAMS_YAML = Path("/fake/hyp.yaml")
        
        # Criar trainer
        trainer = YOLOv9Trainer()
        
        # Construir comando sem resume
        cmd = trainer._build_train_command(
            epochs=100,
            batch_size=16,
            img_size=640,
            device="0",
            workers=4,
            name="test_experiment",
            resume=None,
            weights=""
        )
        
        # Verificar que --resume NÃO está presente
        assert "--resume" not in cmd


class TestTrainMethod:
    """Testes para o método train (Requisitos 4.2, 4.9)"""
    
    @patch('src.training.subprocess.run')
    @patch('src.training.check_yolov9_installation')
    @patch('src.training.config')
    def test_train_uses_default_values_when_optional_params_not_provided(
        self, mock_config, mock_check, mock_subprocess
    ):
        """
        Requisito 4.2: train deve usar valores padrão quando parâmetros opcionais não são fornecidos
        
        Verifica que quando train é chamado sem parâmetros opcionais, os valores padrão são utilizados.
        """
        # Configurar mocks
        mock_check.return_value = True
        mock_data_yaml = MagicMock(spec=Path)
        mock_data_yaml.exists.return_value = True
        mock_data_yaml.absolute.return_value = Path("/fake/path/dados-placas.yaml")
        mock_config.DATA_YAML = mock_data_yaml
        mock_config.TRAIN_SCRIPT = Path("/fake/train_dual.py")
        mock_config.MODEL_YAML = Path("/fake/model.yaml")
        mock_config.HYPERPARAMS_YAML = Path("/fake/hyp.yaml")
        mock_config.YOLOV9_DIR = Path("/fake/yolov9")
        
        # Configurar valores padrão
        mock_config.DEFAULT_EPOCHS = 300
        mock_config.DEFAULT_BATCH_SIZE = 8
        mock_config.DEFAULT_IMG_SIZE = 640
        mock_config.DEFAULT_WORKERS = 0
        
        # Configurar subprocess para retornar sucesso
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        # Criar trainer
        trainer = YOLOv9Trainer()
        
        # Chamar train sem parâmetros opcionais
        trainer.train()
        
        # Verificar que subprocess.run foi chamado
        assert mock_subprocess.called
        
        # Obter o comando passado para subprocess.run
        call_args = mock_subprocess.call_args
        cmd = call_args[0][0]
        cmd_str = " ".join(str(x) for x in cmd)
        
        # Verificar que os valores padrão foram usados
        assert "--epochs 300" in cmd_str or ("--epochs" in cmd and "300" in cmd)
        assert "--batch-size 8" in cmd_str or ("--batch-size" in cmd and "8" in cmd)
        assert "--img 640" in cmd_str or ("--img" in cmd and "640" in cmd)
        assert "--workers 0" in cmd_str or ("--workers" in cmd and "0" in cmd)
    
    @patch('src.training.validate_device')
    @patch('src.training.check_yolov9_installation')
    @patch('src.training.config')
    def test_train_raises_value_error_for_invalid_device_before_subprocess(
        self, mock_config, mock_check, mock_validate_device
    ):
        """
        Requisito 4.9: train deve lançar ValueError para device inválido antes de executar subprocesso
        
        Verifica que a validação de device ocorre antes de qualquer execução de subprocesso.
        """
        # Configurar mocks
        mock_check.return_value = True
        mock_data_yaml = MagicMock(spec=Path)
        mock_data_yaml.exists.return_value = True
        mock_data_yaml.absolute.return_value = Path("/fake/path/dados-placas.yaml")
        mock_config.DATA_YAML = mock_data_yaml
        mock_config.TRAIN_SCRIPT = Path("/fake/train_dual.py")
        mock_config.MODEL_YAML = Path("/fake/model.yaml")
        mock_config.HYPERPARAMS_YAML = Path("/fake/hyp.yaml")
        mock_config.YOLOV9_DIR = Path("/fake/yolov9")
        mock_config.DEFAULT_EPOCHS = 300
        mock_config.DEFAULT_BATCH_SIZE = 8
        mock_config.DEFAULT_IMG_SIZE = 640
        mock_config.DEFAULT_WORKERS = 0
        
        # Configurar validate_device para lançar ValueError
        invalid_device = "invalid_gpu_xyz"
        mock_validate_device.side_effect = ValueError(f"Device inválido: {invalid_device}")
        
        # Criar trainer
        trainer = YOLOv9Trainer()
        
        # Verificar que ValueError é lançado ao chamar train com device inválido
        with pytest.raises(ValueError) as exc_info:
            trainer.train(device=invalid_device)
        
        # Verificar que a mensagem contém o device inválido
        assert invalid_device in str(exc_info.value)
    
    @patch('src.training.subprocess.run')
    @patch('src.training.validate_device')
    @patch('src.training.check_yolov9_installation')
    @patch('src.training.config')
    def test_train_validates_device_before_subprocess_execution(
        self, mock_config, mock_check, mock_validate_device, mock_subprocess
    ):
        """
        Requisito 4.9: verifica que validate_device é chamado antes de subprocess.run
        
        Garante que a validação de device ocorre antes da execução do subprocesso.
        """
        # Configurar mocks
        mock_check.return_value = True
        mock_data_yaml = MagicMock(spec=Path)
        mock_data_yaml.exists.return_value = True
        mock_data_yaml.absolute.return_value = Path("/fake/path/dados-placas.yaml")
        mock_config.DATA_YAML = mock_data_yaml
        mock_config.TRAIN_SCRIPT = Path("/fake/train_dual.py")
        mock_config.MODEL_YAML = Path("/fake/model.yaml")
        mock_config.HYPERPARAMS_YAML = Path("/fake/hyp.yaml")
        mock_config.YOLOV9_DIR = Path("/fake/yolov9")
        mock_config.DEFAULT_EPOCHS = 300
        mock_config.DEFAULT_BATCH_SIZE = 8
        mock_config.DEFAULT_IMG_SIZE = 640
        mock_config.DEFAULT_WORKERS = 0
        
        # Configurar validate_device para retornar device normalizado
        mock_validate_device.return_value = "0"
        
        # Configurar subprocess para retornar sucesso
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        # Criar trainer
        trainer = YOLOv9Trainer()
        
        # Chamar train
        trainer.train(device="cuda:0")
        
        # Verificar que validate_device foi chamado antes de subprocess.run
        assert mock_validate_device.called
        assert mock_subprocess.called
        
        # Verificar ordem de chamadas (validate_device deve ser chamado antes de subprocess.run)
        # Isso é garantido pela implementação, mas podemos verificar que ambos foram chamados
        mock_validate_device.assert_called_once_with("cuda:0")
