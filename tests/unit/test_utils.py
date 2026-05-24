"""
Testes unitários para o módulo utils.

Valida comportamento de funções utilitárias incluindo:
- validate_device (Requisitos 2.2, 2.3)
- check_yolov9_installation (Requisito 2.4)
- parse_yolo_label (Requisitos 2.5, 2.6)
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.utils import validate_device, check_yolov9_installation, parse_yolo_label


class TestValidateDevice:
    """Testes para validate_device (Requisitos 2.2, 2.3)"""
    
    def test_validate_device_cpu(self):
        """Requisito 2.2: 'cpu' deve retornar 'cpu'"""
        assert validate_device('cpu') == 'cpu'
        assert validate_device('CPU') == 'cpu'
        assert validate_device('  cpu  ') == 'cpu'
    
    def test_validate_device_digit(self):
        """Requisito 2.2: dígito numérico deve retornar o próprio dígito"""
        assert validate_device('0') == '0'
        assert validate_device('1') == '1'
        assert validate_device('9') == '9'
    
    def test_validate_device_cuda_format(self):
        """Requisito 2.2: 'cuda:N' deve retornar apenas 'N'"""
        assert validate_device('cuda:0') == '0'
        assert validate_device('cuda:1') == '1'
        assert validate_device('cuda:9') == '9'
        assert validate_device('CUDA:0') == '0'
    
    def test_validate_device_invalid_raises_value_error(self):
        """Requisito 2.3: entrada inválida deve lançar ValueError"""
        with pytest.raises(ValueError):
            validate_device('gpu')
        
        with pytest.raises(ValueError):
            validate_device('cuda')
        
        with pytest.raises(ValueError):
            validate_device('cuda:')
        
        with pytest.raises(ValueError):
            validate_device('')
        
        with pytest.raises(ValueError):
            validate_device('abc')
        
        with pytest.raises(ValueError):
            validate_device('10')  # Dois dígitos
        
        with pytest.raises(ValueError):
            validate_device('cuda:10')  # Dois dígitos após cuda:
    
    def test_validate_device_error_message_contains_invalid_value(self):
        """Requisito 2.3: mensagem de erro deve conter o valor inválido recebido"""
        invalid_value = 'invalid_device_xyz'
        
        with pytest.raises(ValueError) as exc_info:
            validate_device(invalid_value)
        
        # Verifica que a mensagem de erro contém o valor inválido
        assert invalid_value in str(exc_info.value)


class TestCheckYolov9Installation:
    """Testes para check_yolov9_installation (Requisito 2.4)"""
    
    def test_check_yolov9_installation_verifies_all_required_paths(self):
        """
        Requisito 2.4: deve verificar YOLOV9_DIR, train_dual.py, val.py e detect.py
        
        Este teste verifica que a função retorna False quando qualquer um dos
        componentes obrigatórios está ausente.
        """
        # A função deve retornar False se qualquer caminho não existir
        # (assumindo que o ambiente de teste não tem YOLOv9 instalado)
        result = check_yolov9_installation()
        
        # O resultado deve ser booleano
        assert isinstance(result, bool)
        
        # Se retornar True, todos os caminhos devem existir
        # Se retornar False, pelo menos um caminho não existe
        from src.config import config
        
        if result:
            # Se retornou True, verificar que todos os caminhos existem
            assert config.YOLOV9_DIR.exists()
            assert config.TRAIN_SCRIPT.exists()
            assert config.VAL_SCRIPT.exists()
            assert config.DETECT_SCRIPT.exists()
        else:
            # Se retornou False, pelo menos um caminho não existe
            paths_exist = [
                config.YOLOV9_DIR.exists(),
                config.TRAIN_SCRIPT.exists(),
                config.VAL_SCRIPT.exists(),
                config.DETECT_SCRIPT.exists()
            ]
            assert not all(paths_exist)
    
    @patch('src.utils.config')
    def test_check_yolov9_installation_returns_false_when_yolov9_dir_not_exists(self, mock_config):
        """
        Requisito 2.4: deve retornar False quando YOLOV9_DIR não existe
        
        Testa com mock de config para simular YOLOV9_DIR inexistente.
        """
        # Configurar mocks para simular YOLOV9_DIR inexistente
        mock_yolov9_dir = MagicMock(spec=Path)
        mock_yolov9_dir.exists.return_value = False
        
        mock_train_script = MagicMock(spec=Path)
        mock_train_script.exists.return_value = True
        
        mock_val_script = MagicMock(spec=Path)
        mock_val_script.exists.return_value = True
        
        mock_detect_script = MagicMock(spec=Path)
        mock_detect_script.exists.return_value = True
        
        mock_config.YOLOV9_DIR = mock_yolov9_dir
        mock_config.TRAIN_SCRIPT = mock_train_script
        mock_config.VAL_SCRIPT = mock_val_script
        mock_config.DETECT_SCRIPT = mock_detect_script
        
        # Executar função
        result = check_yolov9_installation()
        
        # Verificar que retorna False quando YOLOV9_DIR não existe
        assert result is False
    
    @patch('src.utils.config')
    def test_check_yolov9_installation_returns_false_when_any_script_missing(self, mock_config):
        """
        Requisito 2.4: deve retornar False quando qualquer script está ausente
        
        Testa com mock de config para simular script ausente.
        """
        # Configurar mocks para simular train_dual.py ausente
        mock_yolov9_dir = MagicMock(spec=Path)
        mock_yolov9_dir.exists.return_value = True
        
        mock_train_script = MagicMock(spec=Path)
        mock_train_script.exists.return_value = False  # Script ausente
        
        mock_val_script = MagicMock(spec=Path)
        mock_val_script.exists.return_value = True
        
        mock_detect_script = MagicMock(spec=Path)
        mock_detect_script.exists.return_value = True
        
        mock_config.YOLOV9_DIR = mock_yolov9_dir
        mock_config.TRAIN_SCRIPT = mock_train_script
        mock_config.VAL_SCRIPT = mock_val_script
        mock_config.DETECT_SCRIPT = mock_detect_script
        
        # Executar função
        result = check_yolov9_installation()
        
        # Verificar que retorna False quando qualquer script está ausente
        assert result is False
    
    @patch('src.utils.config')
    def test_check_yolov9_installation_returns_true_when_all_paths_exist(self, mock_config):
        """
        Requisito 2.4: deve retornar True quando todos os caminhos existem
        
        Testa com mock de config para simular instalação válida.
        """
        # Configurar mocks para simular todos os caminhos existentes
        mock_yolov9_dir = MagicMock(spec=Path)
        mock_yolov9_dir.exists.return_value = True
        
        mock_train_script = MagicMock(spec=Path)
        mock_train_script.exists.return_value = True
        
        mock_val_script = MagicMock(spec=Path)
        mock_val_script.exists.return_value = True
        
        mock_detect_script = MagicMock(spec=Path)
        mock_detect_script.exists.return_value = True
        
        mock_config.YOLOV9_DIR = mock_yolov9_dir
        mock_config.TRAIN_SCRIPT = mock_train_script
        mock_config.VAL_SCRIPT = mock_val_script
        mock_config.DETECT_SCRIPT = mock_detect_script
        
        # Executar função
        result = check_yolov9_installation()
        
        # Verificar que retorna True quando todos os caminhos existem
        assert result is True


class TestParseYoloLabel:
    """Testes para parse_yolo_label (Requisitos 2.5, 2.6)"""
    
    def test_parse_yolo_label_valid_file(self):
        """Requisito 2.5: deve retornar lista de dicionários para linhas válidas"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("0 0.5 0.5 0.1 0.1\n")
            f.write("1 0.3 0.4 0.2 0.15\n")
            f.write("34 0.8 0.9 0.05 0.05\n")
            temp_path = Path(f.name)
        
        try:
            result = parse_yolo_label(temp_path)
            
            assert len(result) == 3
            
            # Primeira detecção
            assert result[0]['class_id'] == 0
            assert result[0]['x_center'] == 0.5
            assert result[0]['y_center'] == 0.5
            assert result[0]['width'] == 0.1
            assert result[0]['height'] == 0.1
            
            # Segunda detecção
            assert result[1]['class_id'] == 1
            assert result[1]['x_center'] == 0.3
            
            # Terceira detecção
            assert result[2]['class_id'] == 34
        finally:
            temp_path.unlink()
    
    def test_parse_yolo_label_ignores_invalid_lines(self):
        """Requisito 2.5: deve ignorar silenciosamente linhas com != 5 campos"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("0 0.5 0.5 0.1 0.1\n")  # Válida (5 campos)
            f.write("1 0.3 0.4 0.2\n")  # Inválida (4 campos)
            f.write("2 0.6 0.7 0.1 0.1 0.95\n")  # Inválida (6 campos)
            f.write("3 0.8 0.9 0.05 0.05\n")  # Válida (5 campos)
            f.write("\n")  # Linha vazia
            f.write("invalid line\n")  # Linha inválida
            temp_path = Path(f.name)
        
        try:
            result = parse_yolo_label(temp_path)
            
            # Deve retornar apenas as 2 linhas válidas
            assert len(result) == 2
            assert result[0]['class_id'] == 0
            assert result[1]['class_id'] == 3
        finally:
            temp_path.unlink()
    
    def test_parse_yolo_label_nonexistent_file(self):
        """Requisito 2.6: deve retornar lista vazia para arquivo inexistente"""
        nonexistent_path = Path('/tmp/nonexistent_label_file_xyz.txt')
        
        result = parse_yolo_label(nonexistent_path)
        
        assert result == []
        assert isinstance(result, list)
    
    def test_parse_yolo_label_empty_file(self):
        """Caso de borda: arquivo vazio deve retornar lista vazia"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            result = parse_yolo_label(temp_path)
            
            assert result == []
            assert isinstance(result, list)
        finally:
            temp_path.unlink()
    
    def test_parse_yolo_label_preserves_data_types(self):
        """Requisito 2.5: deve preservar tipos de dados corretos"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("10 0.123456 0.789012 0.345678 0.901234\n")
            temp_path = Path(f.name)
        
        try:
            result = parse_yolo_label(temp_path)
            
            assert len(result) == 1
            detection = result[0]
            
            # class_id deve ser int
            assert isinstance(detection['class_id'], int)
            assert detection['class_id'] == 10
            
            # Demais campos devem ser float
            assert isinstance(detection['x_center'], float)
            assert isinstance(detection['y_center'], float)
            assert isinstance(detection['width'], float)
            assert isinstance(detection['height'], float)
            
            # Verificar valores
            assert detection['x_center'] == pytest.approx(0.123456)
            assert detection['y_center'] == pytest.approx(0.789012)
            assert detection['width'] == pytest.approx(0.345678)
            assert detection['height'] == pytest.approx(0.901234)
        finally:
            temp_path.unlink()
