"""
Testes unitários para o módulo config.py

Valida a configuração centralizada do projeto, incluindo:
- Estrutura de caminhos
- Constantes de classes
- Métodos de validação e utilitários
"""

import pytest
from pathlib import Path
from src.config import config, ProjectConfig


class TestProjectConfigPaths:
    """Testes para validação de caminhos do ProjectConfig."""
    
    def test_root_dir_exists(self):
        """Verifica que ROOT_DIR aponta para o diretório raiz do projeto."""
        assert config.ROOT_DIR.exists()
        assert config.ROOT_DIR.is_dir()
    
    def test_all_paths_are_path_objects(self):
        """Verifica que todos os atributos de caminho são objetos Path."""
        path_attrs = [
            'ROOT_DIR', 'SRC_DIR', 'CONFIGS_DIR', 'DATA_DIR', 'WEIGHTS_DIR',
            'YOLOV9_DIR', 'DATA_YAML', 'HYPERPARAMS_YAML', 'MODEL_YAML',
            'TRAIN_DIR', 'VAL_DIR', 'TEST_DIR', 'TRAIN_SCRIPT', 'VAL_SCRIPT',
            'DETECT_SCRIPT'
        ]
        
        for attr in path_attrs:
            value = getattr(config, attr)
            assert isinstance(value, Path), f"{attr} deve ser um objeto Path"


class TestProjectConfigClassNames:
    """Testes para validação de CLASS_NAMES - Requisito 1.2."""
    
    def test_class_names_count(self):
        """Verifica que CLASS_NAMES tem exatamente 35 elementos."""
        assert len(config.CLASS_NAMES) == 35, \
            f"CLASS_NAMES deve ter 35 elementos, mas tem {len(config.CLASS_NAMES)}"
    
    def test_class_names_order(self):
        """Verifica que CLASS_NAMES está na ordem correta: dígitos 0-9, letras A-Z exceto O."""
        expected_digits = [str(i) for i in range(10)]
        expected_letters = [chr(i) for i in range(ord('A'), ord('Z') + 1) if chr(i) != 'O']
        expected_classes = expected_digits + expected_letters
        
        assert config.CLASS_NAMES == expected_classes, \
            "CLASS_NAMES deve conter dígitos 0-9 seguidos de letras A-Z (exceto O)"
    
    def test_class_names_no_letter_o(self):
        """Verifica que a letra 'O' não está presente em CLASS_NAMES."""
        assert 'O' not in config.CLASS_NAMES, \
            "A letra 'O' não deve estar presente em CLASS_NAMES"
    
    def test_class_names_has_all_digits(self):
        """Verifica que todos os dígitos 0-9 estão presentes."""
        digits = [str(i) for i in range(10)]
        for digit in digits:
            assert digit in config.CLASS_NAMES, \
                f"Dígito '{digit}' deve estar presente em CLASS_NAMES"
    
    def test_class_names_has_correct_letters(self):
        """Verifica que todas as letras corretas (A-Z exceto O) estão presentes."""
        expected_letters = [chr(i) for i in range(ord('A'), ord('Z') + 1) if chr(i) != 'O']
        
        for letter in expected_letters:
            assert letter in config.CLASS_NAMES, \
                f"Letra '{letter}' deve estar presente em CLASS_NAMES"
    
    def test_num_classes_matches_class_names_length(self):
        """Verifica que NUM_CLASSES corresponde ao tamanho de CLASS_NAMES."""
        assert config.NUM_CLASSES == len(config.CLASS_NAMES), \
            "NUM_CLASSES deve ser igual ao tamanho de CLASS_NAMES"


class TestProjectConfigValidatePaths:
    """Testes para o método validate_paths - Requisito 1.7."""
    
    def test_validate_paths_returns_dict(self):
        """Verifica que validate_paths retorna um dicionário."""
        result = config.validate_paths()
        assert isinstance(result, dict)
    
    def test_validate_paths_has_exactly_seven_keys(self):
        """Verifica que validate_paths retorna exatamente 7 chaves."""
        result = config.validate_paths()
        assert len(result) == 7, \
            f"validate_paths deve retornar 7 chaves, mas retornou {len(result)}"
    
    def test_validate_paths_has_correct_keys(self):
        """Verifica que validate_paths retorna as chaves especificadas no Requisito 1.7."""
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
        
        assert actual_keys == expected_keys, \
            f"validate_paths deve retornar as chaves: {expected_keys}, mas retornou: {actual_keys}"
    
    def test_validate_paths_values_are_boolean(self):
        """Verifica que todos os valores retornados são booleanos."""
        result = config.validate_paths()
        
        for key, value in result.items():
            assert isinstance(value, bool), \
                f"O valor para '{key}' deve ser booleano, mas é {type(value)}"
    
    def test_validate_paths_checks_yolov9_directory(self):
        """Verifica que validate_paths verifica o diretório YOLOv9."""
        result = config.validate_paths()
        assert "YOLOv9 Directory" in result
        assert result["YOLOv9 Directory"] == config.YOLOV9_DIR.exists()
    
    def test_validate_paths_checks_all_scripts(self):
        """Verifica que validate_paths verifica todos os scripts."""
        result = config.validate_paths()
        
        script_checks = {
            "Train Script": config.TRAIN_SCRIPT,
            "Validation Script": config.VAL_SCRIPT,
            "Detection Script": config.DETECT_SCRIPT
        }
        
        for key, path in script_checks.items():
            assert key in result
            assert result[key] == path.exists()
    
    def test_validate_paths_checks_all_yaml_files(self):
        """Verifica que validate_paths verifica todos os arquivos YAML."""
        result = config.validate_paths()
        
        yaml_checks = {
            "Data YAML": config.DATA_YAML,
            "Hyperparams YAML": config.HYPERPARAMS_YAML,
            "Model YAML": config.MODEL_YAML
        }
        
        for key, path in yaml_checks.items():
            assert key in result
            assert result[key] == path.exists()


class TestProjectConfigGetTimestamp:
    """Testes para o método get_timestamp - usado em data_preparation.py."""
    
    def test_get_timestamp_exists(self):
        """Verifica que o método get_timestamp existe."""
        assert hasattr(config, 'get_timestamp'), \
            "ProjectConfig deve ter o método get_timestamp"
    
    def test_get_timestamp_returns_string(self):
        """Verifica que get_timestamp retorna uma string."""
        timestamp = config.get_timestamp()
        assert isinstance(timestamp, str), \
            f"get_timestamp deve retornar string, mas retornou {type(timestamp)}"
    
    def test_get_timestamp_format(self):
        """Verifica que get_timestamp retorna string no formato YYYYMMDD_HHMMSS."""
        timestamp = config.get_timestamp()
        
        # Formato esperado: YYYYMMDD_HHMMSS (15 caracteres)
        assert len(timestamp) == 15, \
            f"Timestamp deve ter 15 caracteres, mas tem {len(timestamp)}"
        
        assert '_' in timestamp, \
            "Timestamp deve conter underscore separando data e hora"
        
        parts = timestamp.split('_')
        assert len(parts) == 2, \
            "Timestamp deve ter exatamente uma parte de data e uma de hora"
        
        date_part, time_part = parts
        assert len(date_part) == 8, \
            f"Parte da data deve ter 8 caracteres (YYYYMMDD), mas tem {len(date_part)}"
        
        assert len(time_part) == 6, \
            f"Parte da hora deve ter 6 caracteres (HHMMSS), mas tem {len(time_part)}"
        
        assert date_part.isdigit(), \
            "Parte da data deve conter apenas dígitos"
        
        assert time_part.isdigit(), \
            "Parte da hora deve conter apenas dígitos"
    
    def test_get_timestamp_is_callable(self):
        """Verifica que get_timestamp pode ser chamado múltiplas vezes."""
        timestamp1 = config.get_timestamp()
        timestamp2 = config.get_timestamp()
        
        # Ambos devem ser strings válidas
        assert isinstance(timestamp1, str)
        assert isinstance(timestamp2, str)
        
        # Podem ser iguais ou diferentes dependendo do timing
        # Apenas verificamos que ambos têm o formato correto
        assert len(timestamp1) == 15
        assert len(timestamp2) == 15


class TestProjectConfigDefaults:
    """Testes para valores padrão de configuração."""
    
    def test_default_training_params(self):
        """Verifica valores padrão de treinamento - Requisito 1.3."""
        assert config.DEFAULT_EPOCHS == 300
        assert config.DEFAULT_BATCH_SIZE == 8
        assert config.DEFAULT_IMG_SIZE == 640
        assert config.DEFAULT_WORKERS == 0
    
    def test_default_detection_params(self):
        """Verifica valores padrão de detecção - Requisito 1.4."""
        assert config.DEFAULT_CONF_THRESHOLD == 0.25
        assert config.DEFAULT_IOU_THRESHOLD == 0.45
    
    def test_default_validation_params(self):
        """Verifica valores padrão de validação - Requisito 1.5."""
        assert config.DEFAULT_VAL_CONF == 0.001
        assert config.DEFAULT_VAL_IOU == 0.7
        assert config.DEFAULT_VAL_BATCH == 32


class TestProjectConfigCreateDirectories:
    """Testes para o método create_directories - Requisito 1.8."""
    
    def test_create_directories_exists(self):
        """Verifica que o método create_directories existe."""
        assert hasattr(config, 'create_directories')
    
    def test_create_directories_is_callable(self):
        """Verifica que create_directories pode ser chamado sem erros."""
        # Deve executar sem lançar exceção
        config.create_directories()
    
    def test_create_directories_creates_data_dirs(self):
        """Verifica que create_directories cria os diretórios de dados."""
        config.create_directories()
        
        # Verifica que os diretórios principais existem
        assert config.DATA_DIR.exists()
        assert config.TRAIN_DIR.exists()
        assert config.VAL_DIR.exists()
        assert config.TEST_DIR.exists()
        assert config.WEIGHTS_DIR.exists()
        assert config.CONFIGS_DIR.exists()
    
    def test_create_directories_no_exception_if_exists(self, tmp_path):
        """
        Verifica que create_directories não lança exceção se diretórios já existirem.
        
        Requisito 1.8: WHEN o método `create_directories` é chamado, 
        THE ProjectConfig SHALL criar os diretórios que não existirem, 
        sem lançar exceção se algum deles já existir.
        """
        # Cria uma instância temporária de ProjectConfig com caminhos no tmp_path
        test_config = ProjectConfig()
        
        # Sobrescreve os caminhos para usar tmp_path
        test_config.DATA_DIR = tmp_path / "dados"
        test_config.TRAIN_DIR = tmp_path / "dados" / "treino"
        test_config.VAL_DIR = tmp_path / "dados" / "validacao"
        test_config.TEST_DIR = tmp_path / "dados" / "teste"
        test_config.WEIGHTS_DIR = tmp_path / "pesos"
        test_config.CONFIGS_DIR = tmp_path / "configs"
        
        # Primeira chamada: cria os diretórios
        test_config.create_directories()
        
        # Verifica que os diretórios foram criados
        assert test_config.DATA_DIR.exists()
        assert test_config.TRAIN_DIR.exists()
        assert test_config.VAL_DIR.exists()
        assert test_config.TEST_DIR.exists()
        assert test_config.WEIGHTS_DIR.exists()
        assert test_config.CONFIGS_DIR.exists()
        
        # Segunda chamada: não deve lançar exceção mesmo que os diretórios já existam
        try:
            test_config.create_directories()
            exception_raised = False
        except Exception as e:
            exception_raised = True
            pytest.fail(f"create_directories lançou exceção quando diretórios já existiam: {e}")
        
        assert not exception_raised, \
            "create_directories não deve lançar exceção quando diretórios já existem"
        
        # Verifica que os diretórios ainda existem após a segunda chamada
        assert test_config.DATA_DIR.exists()
        assert test_config.TRAIN_DIR.exists()
        assert test_config.VAL_DIR.exists()
        assert test_config.TEST_DIR.exists()
        assert test_config.WEIGHTS_DIR.exists()
        assert test_config.CONFIGS_DIR.exists()


class TestProjectConfigEnvironment:
    """Testes para configuração de ambiente."""
    
    def test_wandb_disabled(self):
        """Verifica que WANDB está desabilitado - Requisito 1.6."""
        import os
        
        assert os.environ.get("WANDB_MODE") == "disabled"
        assert os.environ.get("WANDB_DISABLED") == "true"
