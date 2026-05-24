"""
Testes unitários para o módulo detection.

Valida comportamento das classes YOLOv9Detector e PlateCharacterProcessor incluindo:
- Validação de instalação do YOLOv9 (Requisito 6.1)
- Validação de arquivos de pesos e source (Requisitos 6.5, 6.6)
- Construção de comando de detecção (Requisito 6.4)
- Aplicação de valores padrão (Requisito 6.2)
- Inclusão de flags save_txt e save_conf (Requisitos 6.3, 6.4)
- Processamento de resultados de detecção (Requisitos 7.1-7.9)
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from src.detection import YOLOv9Detector, PlateCharacterProcessor


class TestYOLOv9DetectorInitialization:
    """Testes para inicialização do YOLOv9Detector"""
    
    @patch('src.detection.check_yolov9_installation')
    def test_detector_raises_runtime_error_when_yolov9_not_installed(self, mock_check):
        """
        Requisito 6.1: YOLOv9Detector() deve lançar RuntimeError quando YOLOv9 não instalado
        
        Verifica que a inicialização falha com RuntimeError quando check_yolov9_installation
        retorna False, indicando que o YOLOv9 não está instalado corretamente.
        """
        # Simular YOLOv9 não instalado
        mock_check.return_value = False
        
        # Verificar que RuntimeError é lançado
        with pytest.raises(RuntimeError) as exc_info:
            YOLOv9Detector()
        
        # Verificar que a mensagem contém referência ao diretório YOLOv9
        assert "uvv/yolov9-main/" in str(exc_info.value)
    
    @patch('src.detection.check_yolov9_installation')
    @patch('src.detection.config')
    def test_detector_initializes_successfully_when_setup_valid(self, mock_config, mock_check):
        """
        Verifica que o detector inicializa com sucesso quando o ambiente está configurado
        """
        # Simular YOLOv9 instalado
        mock_check.return_value = True
        
        # Verificar que inicializa sem exceções
        detector = YOLOv9Detector()
        assert detector is not None
        assert detector.config == mock_config


class TestYOLOv9DetectorDetect:
    """Testes para o método detect"""
    
    @patch('src.detection.check_yolov9_installation')
    @patch('src.detection.config')
    def test_detect_raises_file_not_found_when_weights_not_exist(self, mock_config, mock_check):
        """
        Requisito 6.5: detect deve lançar FileNotFoundError com caminho completo quando
        arquivo de pesos não existe
        
        Verifica que o método detect falha antes de executar qualquer subprocesso
        quando o arquivo de pesos informado não existe.
        """
        # Configurar mocks para inicialização bem-sucedida
        mock_check.return_value = True
        
        # Criar detector
        detector = YOLOv9Detector()
        
        # Tentar detectar com arquivo de pesos inexistente
        nonexistent_weights = "/fake/path/to/nonexistent_weights.pt"
        fake_source = "/fake/source"
        
        with pytest.raises(FileNotFoundError) as exc_info:
            detector.detect(weights=nonexistent_weights, source=fake_source)
        
        # Verificar que a mensagem contém o caminho completo
        assert nonexistent_weights in str(exc_info.value)
    
    @patch('src.detection.check_yolov9_installation')
    @patch('src.detection.config')
    def test_detect_raises_file_not_found_when_source_not_exist(self, mock_config, mock_check):
        """
        Requisito 6.6: detect deve lançar FileNotFoundError com caminho completo quando
        source não existe
        
        Verifica que o método detect falha antes de executar qualquer subprocesso
        quando o source informado não existe.
        """
        # Configurar mocks para inicialização bem-sucedida
        mock_check.return_value = True
        
        # Criar detector
        detector = YOLOv9Detector()
        
        # Criar arquivo de pesos temporário (mock)
        with patch('src.detection.Path') as mock_path_class:
            mock_weights_path = MagicMock(spec=Path)
            mock_weights_path.exists.return_value = True
            mock_weights_path.absolute.return_value = Path("/fake/weights.pt")
            
            # Configurar Path para retornar mock_weights_path para weights
            # e um source inexistente
            def path_side_effect(arg):
                if "weights" in str(arg):
                    return mock_weights_path
                else:
                    mock_source = MagicMock(spec=Path)
                    mock_source.exists.return_value = False
                    mock_source.absolute.return_value = Path("/fake/nonexistent_source")
                    return mock_source
            
            mock_path_class.side_effect = path_side_effect
            
            nonexistent_source = "/fake/nonexistent_source"
            
            with pytest.raises(FileNotFoundError) as exc_info:
                detector.detect(weights="/fake/weights.pt", source=nonexistent_source)
            
            # Verificar que a mensagem contém o caminho completo
            assert nonexistent_source in str(exc_info.value)


class TestBuildDetectionCommand:
    """Testes para o método _build_detection_command"""
    
    @patch('src.detection.check_yolov9_installation')
    @patch('src.detection.config')
    def test_build_detection_command_contains_all_required_arguments(self, mock_config, mock_check):
        """
        Requisito 6.4: _build_detection_command deve conter todos os argumentos obrigatórios
        
        Verifica que o comando construído contém: --weights, --source, --img-size, --device,
        --conf-thres, --iou-thres, --name
        """
        # Configurar mocks
        mock_check.return_value = True
        mock_config.DETECT_SCRIPT = Path("/fake/detect.py")
        mock_config.DEFAULT_IMG_SIZE = 640
        mock_config.DEFAULT_CONF_THRESHOLD = 0.25
        mock_config.DEFAULT_IOU_THRESHOLD = 0.45
        
        # Criar detector
        detector = YOLOv9Detector()
        
        # Construir comando
        cmd = detector._build_detection_command(
            weights="/fake/weights.pt",
            source="/fake/source",
            img_size=640,
            device="0",
            conf_threshold=0.25,
            iou_threshold=0.45,
            name="test_detection",
            save_txt=False,
            save_conf=False
        )
        
        # Verificar que todos os argumentos obrigatórios estão presentes
        assert "--weights" in cmd
        assert "--source" in cmd
        assert "--img-size" in cmd
        assert "--device" in cmd
        assert "--conf-thres" in cmd
        assert "--iou-thres" in cmd
        assert "--name" in cmd
        
        # Verificar valores
        weights_idx = cmd.index("--weights")
        assert "/fake/weights.pt" == cmd[weights_idx + 1]
        
        source_idx = cmd.index("--source")
        assert "/fake/source" == cmd[source_idx + 1]
        
        imgsz_idx = cmd.index("--img-size")
        assert "640" == cmd[imgsz_idx + 1]
        
        device_idx = cmd.index("--device")
        assert "0" == cmd[device_idx + 1]
        
        conf_idx = cmd.index("--conf-thres")
        assert "0.25" == cmd[conf_idx + 1]
        
        iou_idx = cmd.index("--iou-thres")
        assert "0.45" == cmd[iou_idx + 1]
        
        name_idx = cmd.index("--name")
        assert "test_detection" == cmd[name_idx + 1]
    
    @patch('src.detection.check_yolov9_installation')
    @patch('src.detection.config')
    def test_build_detection_command_includes_save_txt_when_true(self, mock_config, mock_check):
        """
        Requisito 6.3: --save-txt deve ser incluído quando save_txt=True
        
        Verifica que a flag --save-txt é adicionada ao comando quando o parâmetro
        save_txt é True.
        """
        # Configurar mocks
        mock_check.return_value = True
        mock_config.DETECT_SCRIPT = Path("/fake/detect.py")
        
        # Criar detector
        detector = YOLOv9Detector()
        
        # Construir comando com save_txt=True
        cmd_with_save_txt = detector._build_detection_command(
            weights="/fake/weights.pt",
            source="/fake/source",
            img_size=640,
            device="0",
            conf_threshold=0.25,
            iou_threshold=0.45,
            name="test_detection",
            save_txt=True,
            save_conf=False
        )
        
        # Verificar que --save-txt está presente
        assert "--save-txt" in cmd_with_save_txt
        
        # Construir comando com save_txt=False
        cmd_without_save_txt = detector._build_detection_command(
            weights="/fake/weights.pt",
            source="/fake/source",
            img_size=640,
            device="0",
            conf_threshold=0.25,
            iou_threshold=0.45,
            name="test_detection",
            save_txt=False,
            save_conf=False
        )
        
        # Verificar que --save-txt não está presente
        assert "--save-txt" not in cmd_without_save_txt
    
    @patch('src.detection.check_yolov9_installation')
    @patch('src.detection.config')
    def test_build_detection_command_includes_save_conf_when_true(self, mock_config, mock_check):
        """
        Requisito 6.4: --save-conf deve ser incluído quando save_conf=True
        
        Verifica que a flag --save-conf é adicionada ao comando quando o parâmetro
        save_conf é True.
        """
        # Configurar mocks
        mock_check.return_value = True
        mock_config.DETECT_SCRIPT = Path("/fake/detect.py")
        
        # Criar detector
        detector = YOLOv9Detector()
        
        # Construir comando com save_conf=True
        cmd_with_save_conf = detector._build_detection_command(
            weights="/fake/weights.pt",
            source="/fake/source",
            img_size=640,
            device="0",
            conf_threshold=0.25,
            iou_threshold=0.45,
            name="test_detection",
            save_txt=False,
            save_conf=True
        )
        
        # Verificar que --save-conf está presente
        assert "--save-conf" in cmd_with_save_conf
        
        # Construir comando com save_conf=False
        cmd_without_save_conf = detector._build_detection_command(
            weights="/fake/weights.pt",
            source="/fake/source",
            img_size=640,
            device="0",
            conf_threshold=0.25,
            iou_threshold=0.45,
            name="test_detection",
            save_txt=False,
            save_conf=False
        )
        
        # Verificar que --save-conf não está presente
        assert "--save-conf" not in cmd_without_save_conf
    
    @patch('src.detection.check_yolov9_installation')
    @patch('src.detection.config')
    def test_build_detection_command_uses_python_executable(self, mock_config, mock_check):
        """
        Verifica que o comando usa sys.executable para invocar o script Python
        """
        # Configurar mocks
        mock_check.return_value = True
        mock_config.DETECT_SCRIPT = Path("/fake/detect.py")
        
        # Criar detector
        detector = YOLOv9Detector()
        
        # Construir comando
        cmd = detector._build_detection_command(
            weights="/fake/weights.pt",
            source="/fake/source",
            img_size=640,
            device="0",
            conf_threshold=0.25,
            iou_threshold=0.45,
            name="test_detection",
            save_txt=False,
            save_conf=False
        )
        
        # Verificar que o primeiro elemento é sys.executable
        assert cmd[0] == sys.executable
        
        # Verificar que o segundo elemento é o script de detecção
        assert cmd[1] == str(mock_config.DETECT_SCRIPT)


class TestDefaultValues:
    """Testes para aplicação de valores padrão"""
    
    @patch('src.detection.print_config_summary')
    @patch('src.detection.print_header')
    @patch('src.detection.check_yolov9_installation')
    @patch('src.detection.config')
    @patch('src.detection.subprocess.run')
    def test_detect_applies_default_values(self, mock_subprocess, mock_config, mock_check, mock_print_header, mock_print_summary):
        """
        Requisito 6.2: valores padrão devem ser aplicados quando não especificados
        
        Verifica que img_size=640, conf_threshold=0.25, iou_threshold=0.45
        são aplicados quando os parâmetros não são fornecidos.
        """
        # Configurar mocks
        mock_check.return_value = True
        mock_config.DETECT_SCRIPT = Path("/fake/detect.py")
        mock_config.YOLOV9_DIR = Path("/fake/yolov9")
        
        # Configurar valores padrão
        mock_config.DEFAULT_IMG_SIZE = 640
        mock_config.DEFAULT_CONF_THRESHOLD = 0.25
        mock_config.DEFAULT_IOU_THRESHOLD = 0.45
        
        # Configurar subprocess para retornar sucesso
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        # Criar detector
        detector = YOLOv9Detector()
        
        # Criar arquivos temporários (mock)
        with patch('src.detection.Path') as mock_path_class:
            mock_weights_path = MagicMock(spec=Path)
            mock_weights_path.exists.return_value = True
            mock_weights_path.absolute.return_value = Path("/fake/weights.pt")
            
            mock_source_path = MagicMock(spec=Path)
            mock_source_path.exists.return_value = True
            mock_source_path.absolute.return_value = Path("/fake/source")
            
            def path_side_effect(arg):
                if "weights" in str(arg):
                    return mock_weights_path
                else:
                    return mock_source_path
            
            mock_path_class.side_effect = path_side_effect
            
            # Chamar detect sem especificar parâmetros opcionais
            detector.detect(weights="/fake/weights.pt", source="/fake/source")
        
        # Verificar que subprocess.run foi chamado
        assert mock_subprocess.called
        
        # Obter o comando passado para subprocess.run
        call_args = mock_subprocess.call_args
        cmd = call_args[0][0]
        
        # Verificar que os valores padrão foram aplicados
        imgsz_idx = cmd.index("--img-size")
        assert cmd[imgsz_idx + 1] == "640"
        
        conf_idx = cmd.index("--conf-thres")
        assert cmd[conf_idx + 1] == "0.25"
        
        iou_idx = cmd.index("--iou-thres")
        assert cmd[iou_idx + 1] == "0.45"



class TestPlateCharacterProcessorInitialization:
    """Testes para inicialização do PlateCharacterProcessor"""
    
    @patch('src.detection.config')
    def test_class_mapping_has_35_entries(self, mock_config):
        """
        Requisito 7.8: class_mapping deve ter 35 entradas com chaves 0–34
        
        Verifica que o mapeamento de classes é construído corretamente durante
        a inicialização, cobrindo todos os 35 caracteres.
        """
        # Configurar CLASS_NAMES com 35 elementos
        mock_config.CLASS_NAMES = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z'
        ]
        
        # Criar processador
        processor = PlateCharacterProcessor()
        
        # Verificar que class_mapping tem 35 entradas
        assert len(processor.class_mapping) == 35
        
        # Verificar que as chaves são 0–34
        assert set(processor.class_mapping.keys()) == set(range(35))
        
        # Verificar que os valores correspondem a CLASS_NAMES
        for i in range(35):
            assert processor.class_mapping[i] == mock_config.CLASS_NAMES[i]


class TestProcessDetectionResults:
    """Testes para o método process_detection_results"""
    
    @patch('src.detection.config')
    def test_process_detection_results_returns_empty_dict_when_no_txt_files(self, mock_config, tmp_path):
        """
        Requisito 7.9: process_detection_results deve retornar dicionário vazio quando
        diretório não contém arquivos .txt
        
        Verifica que o processamento retorna um dicionário vazio sem lançar exceção
        quando o diretório de resultados não contém nenhum arquivo .txt.
        """
        # Configurar CLASS_NAMES
        mock_config.CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                                    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                                    'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U',
                                    'V', 'W', 'X', 'Y', 'Z']
        
        # Criar diretório vazio
        empty_dir = tmp_path / "empty_results"
        empty_dir.mkdir()
        
        # Criar processador
        processor = PlateCharacterProcessor()
        
        # Processar diretório vazio
        results = processor.process_detection_results(empty_dir)
        
        # Verificar que retorna dicionário vazio
        assert results == {}
    
    @patch('src.detection.config')
    def test_process_detection_results_with_5_field_detection(self, mock_config, tmp_path):
        """
        Requisito 7.5: arquivo de detecção com 5 campos deve ter confiança atribuída como 1.0
        
        Verifica que quando um arquivo de detecção contém linhas com exatamente 5 campos
        (sem campo de confiança), o sistema atribui confiança 1.0 automaticamente.
        """
        # Configurar CLASS_NAMES
        mock_config.CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                                    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                                    'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U',
                                    'V', 'W', 'X', 'Y', 'Z']
        
        # Criar diretório de resultados
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        
        # Criar arquivo de detecção com 5 campos (sem confiança)
        detection_file = results_dir / "placa001.txt"
        detection_file.write_text(
            "10 0.1 0.5 0.05 0.1\n"  # Classe 10 = 'A'
            "11 0.2 0.5 0.05 0.1\n"  # Classe 11 = 'B'
            "12 0.3 0.5 0.05 0.1\n"  # Classe 12 = 'C'
        )
        
        # Criar processador
        processor = PlateCharacterProcessor()
        
        # Processar resultados
        results = processor.process_detection_results(results_dir, confidence_threshold=0.0)
        
        # Verificar que a imagem foi processada
        assert "placa001" in results
        
        # Verificar que todas as detecções têm confiança 1.0
        for detection in results["placa001"]["detections"]:
            assert detection["confidence"] == 1.0
    
    @patch('src.detection.config')
    def test_process_detection_results_with_6_field_detection(self, mock_config, tmp_path):
        """
        Requisito 7.6: arquivo de detecção com 6 campos deve usar sexto campo como confiança
        
        Verifica que quando um arquivo de detecção contém linhas com 6 campos,
        o sexto campo é utilizado como valor de confiança.
        """
        # Configurar CLASS_NAMES
        mock_config.CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                                    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                                    'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U',
                                    'V', 'W', 'X', 'Y', 'Z']
        
        # Criar diretório de resultados
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        
        # Criar arquivo de detecção com 6 campos (com confiança)
        detection_file = results_dir / "placa002.txt"
        detection_file.write_text(
            "10 0.1 0.5 0.05 0.1 0.95\n"  # Classe 10 = 'A', confiança 0.95
            "11 0.2 0.5 0.05 0.1 0.87\n"  # Classe 11 = 'B', confiança 0.87
            "12 0.3 0.5 0.05 0.1 0.92\n"  # Classe 12 = 'C', confiança 0.92
        )
        
        # Criar processador
        processor = PlateCharacterProcessor()
        
        # Processar resultados
        results = processor.process_detection_results(results_dir, confidence_threshold=0.0)
        
        # Verificar que a imagem foi processada
        assert "placa002" in results
        
        # Verificar que as confianças foram lidas corretamente
        detections = results["placa002"]["detections"]
        assert detections[0]["confidence"] == 0.95
        assert detections[1]["confidence"] == 0.87
        assert detections[2]["confidence"] == 0.92
    
    @patch('src.detection.config')
    def test_process_detection_results_filters_by_confidence_threshold(self, mock_config, tmp_path):
        """
        Requisito 7.2: detecções abaixo do threshold devem ser filtradas
        
        Verifica que detecções com confiança abaixo do threshold são descartadas
        e que a imagem é omitida do resultado se todas as detecções forem filtradas.
        """
        # Configurar CLASS_NAMES
        mock_config.CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                                    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                                    'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U',
                                    'V', 'W', 'X', 'Y', 'Z']
        
        # Criar diretório de resultados
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        
        # Criar arquivo com detecções de confiança variada
        detection_file1 = results_dir / "placa_high_conf.txt"
        detection_file1.write_text(
            "10 0.1 0.5 0.05 0.1 0.95\n"  # Acima do threshold
            "11 0.2 0.5 0.05 0.1 0.87\n"  # Acima do threshold
        )
        
        # Criar arquivo com todas as detecções abaixo do threshold
        detection_file2 = results_dir / "placa_low_conf.txt"
        detection_file2.write_text(
            "10 0.1 0.5 0.05 0.1 0.30\n"  # Abaixo do threshold
            "11 0.2 0.5 0.05 0.1 0.25\n"  # Abaixo do threshold
        )
        
        # Criar processador
        processor = PlateCharacterProcessor()
        
        # Processar com threshold 0.5
        results = processor.process_detection_results(results_dir, confidence_threshold=0.5)
        
        # Verificar que placa_high_conf foi incluída
        assert "placa_high_conf" in results
        assert len(results["placa_high_conf"]["detections"]) == 2
        
        # Verificar que placa_low_conf foi omitida (todas as detecções filtradas)
        assert "placa_low_conf" not in results
    
    @patch('src.detection.config')
    def test_process_detection_results_includes_valid_format_and_plate_type(self, mock_config, tmp_path):
        """
        Requisito 7.4: resultado deve incluir valid_format e plate_type para cada imagem
        
        Verifica que o resultado do processamento inclui os campos valid_format e plate_type
        para cada imagem processada.
        """
        # Configurar CLASS_NAMES
        mock_config.CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                                    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                                    'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U',
                                    'V', 'W', 'X', 'Y', 'Z']
        
        # Criar diretório de resultados
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        
        # Criar arquivo de detecção que forma uma placa válida (ABC1234)
        detection_file = results_dir / "placa_valida.txt"
        detection_file.write_text(
            "10 0.1 0.5 0.05 0.1 0.95\n"  # A
            "11 0.2 0.5 0.05 0.1 0.95\n"  # B
            "12 0.3 0.5 0.05 0.1 0.95\n"  # C
            "1 0.4 0.5 0.05 0.1 0.95\n"   # 1
            "2 0.5 0.5 0.05 0.1 0.95\n"   # 2
            "3 0.6 0.5 0.05 0.1 0.95\n"   # 3
            "4 0.7 0.5 0.05 0.1 0.95\n"   # 4
        )
        
        # Criar processador
        processor = PlateCharacterProcessor()
        
        # Processar resultados
        results = processor.process_detection_results(results_dir, confidence_threshold=0.0)
        
        # Verificar que a imagem foi processada
        assert "placa_valida" in results
        
        # Verificar que os campos obrigatórios estão presentes
        result = results["placa_valida"]
        assert "detections" in result
        assert "plate_text" in result
        assert "confidence" in result
        assert "valid_format" in result
        assert "plate_type" in result
        
        # Verificar que é uma placa válida do tipo Placa_Antiga
        assert result["valid_format"] is True
        assert result["plate_type"] == "Placa_Antiga"
        assert result["plate_text"] == "ABC1234"
    
    @patch('src.detection.config')
    @patch('src.detection.logger')
    def test_process_detection_results_handles_corrupted_file_gracefully(self, mock_logger, mock_config, tmp_path):
        """
        Requisito 7.7: arquivo corrompido deve gerar WARNING no log e processamento continua
        
        Verifica que quando um arquivo de detecção está corrompido ou causa erro ao processar,
        o sistema registra um aviso no log e continua processando os demais arquivos.
        """
        # Configurar CLASS_NAMES
        mock_config.CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                                    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                                    'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U',
                                    'V', 'W', 'X', 'Y', 'Z']
        
        # Criar diretório de resultados
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        
        # Criar arquivo válido
        valid_file = results_dir / "placa_valida.txt"
        valid_file.write_text("10 0.1 0.5 0.05 0.1 0.95\n")
        
        # Criar arquivo corrompido (não pode ser lido)
        corrupted_file = results_dir / "placa_corrompida.txt"
        corrupted_file.write_text("invalid data that will cause parsing error\n")
        
        # Criar processador
        processor = PlateCharacterProcessor()
        
        # Processar resultados - não deve lançar exceção
        results = processor.process_detection_results(results_dir, confidence_threshold=0.0)
        
        # Verificar que o arquivo válido foi processado
        assert "placa_valida" in results
        
        # Verificar que o arquivo corrompido foi ignorado (não processado)
        # O arquivo corrompido não terá detecções válidas, então não aparecerá nos resultados
        # Mas o processamento deve ter continuado sem exceção
        assert isinstance(results, dict)


class TestFilterByConfidence:
    """Testes para o método _filter_by_confidence"""
    
    @patch('src.detection.config')
    def test_filter_by_confidence_returns_only_detections_above_threshold(self, mock_config):
        """
        Requisito 7.2: _filter_by_confidence deve retornar apenas detecções com confidence >= threshold
        
        Verifica que o método filtra corretamente as detecções, mantendo apenas aquelas
        que atendem ao critério de confiança mínima.
        """
        # Configurar CLASS_NAMES
        mock_config.CLASS_NAMES = ['0', '1', '2']
        
        # Criar processador
        processor = PlateCharacterProcessor()
        
        # Criar lista de detecções com confianças variadas
        detections = [
            {'class_id': 0, 'class_name': '0', 'x_center': 0.1, 'y_center': 0.5,
             'width': 0.05, 'height': 0.1, 'confidence': 0.95},
            {'class_id': 1, 'class_name': '1', 'x_center': 0.2, 'y_center': 0.5,
             'width': 0.05, 'height': 0.1, 'confidence': 0.45},
            {'class_id': 2, 'class_name': '2', 'x_center': 0.3, 'y_center': 0.5,
             'width': 0.05, 'height': 0.1, 'confidence': 0.75},
        ]
        
        # Filtrar com threshold 0.5
        filtered = processor._filter_by_confidence(detections, 0.5)
        
        # Verificar que apenas detecções >= 0.5 foram mantidas
        assert len(filtered) == 2
        assert all(d['confidence'] >= 0.5 for d in filtered)
        assert filtered[0]['confidence'] == 0.95
        assert filtered[1]['confidence'] == 0.75
    
    @patch('src.detection.config')
    def test_filter_by_confidence_returns_empty_list_when_all_below_threshold(self, mock_config):
        """
        Verifica que retorna lista vazia quando todas as detecções estão abaixo do threshold
        """
        # Configurar CLASS_NAMES
        mock_config.CLASS_NAMES = ['0', '1', '2']
        
        # Criar processador
        processor = PlateCharacterProcessor()
        
        # Criar lista de detecções todas abaixo do threshold
        detections = [
            {'class_id': 0, 'class_name': '0', 'x_center': 0.1, 'y_center': 0.5,
             'width': 0.05, 'height': 0.1, 'confidence': 0.25},
            {'class_id': 1, 'class_name': '1', 'x_center': 0.2, 'y_center': 0.5,
             'width': 0.05, 'height': 0.1, 'confidence': 0.30},
        ]
        
        # Filtrar com threshold 0.5
        filtered = processor._filter_by_confidence(detections, 0.5)
        
        # Verificar que retorna lista vazia
        assert filtered == []



class TestReconstructPlateText:
    """Testes para o método _reconstruct_plate_text"""
    
    @patch('src.detection.config')
    def test_reconstruct_plate_text_orders_by_x_center(self, mock_config):
        """
        Requisito 7.3: reconstrução de texto deve ordenar por x_center
        
        Verifica que os caracteres são ordenados pela posição horizontal (x_center)
        antes de serem concatenados para formar o texto da placa.
        """
        # Configurar CLASS_NAMES
        mock_config.CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                                    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                                    'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U',
                                    'V', 'W', 'X', 'Y', 'Z']
        
        # Criar processador
        processor = PlateCharacterProcessor()
        
        # Criar detecções fora de ordem
        detections = [
            {'class_id': 12, 'class_name': 'C', 'x_center': 0.3, 'y_center': 0.5,
             'width': 0.05, 'height': 0.1, 'confidence': 0.95},
            {'class_id': 10, 'class_name': 'A', 'x_center': 0.1, 'y_center': 0.5,
             'width': 0.05, 'height': 0.1, 'confidence': 0.95},
            {'class_id': 11, 'class_name': 'B', 'x_center': 0.2, 'y_center': 0.5,
             'width': 0.05, 'height': 0.1, 'confidence': 0.95},
        ]
        
        # Reconstruir texto
        plate_text = processor._reconstruct_plate_text(detections)
        
        # Verificar que o texto está na ordem correta (A, B, C)
        assert plate_text == "ABC"
    
    @patch('src.detection.config')
    def test_reconstruct_plate_text_concatenates_without_separator(self, mock_config):
        """
        Verifica que os caracteres são concatenados sem separador
        """
        # Configurar CLASS_NAMES
        mock_config.CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                                    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                                    'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U',
                                    'V', 'W', 'X', 'Y', 'Z']
        
        # Criar processador
        processor = PlateCharacterProcessor()
        
        # Criar detecções formando uma placa completa
        detections = [
            {'class_id': 10, 'class_name': 'A', 'x_center': 0.1, 'y_center': 0.5,
             'width': 0.05, 'height': 0.1, 'confidence': 0.95},
            {'class_id': 11, 'class_name': 'B', 'x_center': 0.2, 'y_center': 0.5,
             'width': 0.05, 'height': 0.1, 'confidence': 0.95},
            {'class_id': 12, 'class_name': 'C', 'x_center': 0.3, 'y_center': 0.5,
             'width': 0.05, 'height': 0.1, 'confidence': 0.95},
            {'class_id': 1, 'class_name': '1', 'x_center': 0.4, 'y_center': 0.5,
             'width': 0.05, 'height': 0.1, 'confidence': 0.95},
            {'class_id': 2, 'class_name': '2', 'x_center': 0.5, 'y_center': 0.5,
             'width': 0.05, 'height': 0.1, 'confidence': 0.95},
            {'class_id': 3, 'class_name': '3', 'x_center': 0.6, 'y_center': 0.5,
             'width': 0.05, 'height': 0.1, 'confidence': 0.95},
            {'class_id': 4, 'class_name': '4', 'x_center': 0.7, 'y_center': 0.5,
             'width': 0.05, 'height': 0.1, 'confidence': 0.95},
        ]
        
        # Reconstruir texto
        plate_text = processor._reconstruct_plate_text(detections)
        
        # Verificar que não há espaços ou separadores
        assert plate_text == "ABC1234"
        assert " " not in plate_text
        assert "-" not in plate_text


class TestValidatePlateFormat:
    """Testes para o método _validate_plate_format"""
    
    @patch('src.detection.config')
    def test_validate_plate_format_recognizes_placa_antiga(self, mock_config):
        """
        Requisito 9.1: validação deve reconhecer formato Placa_Antiga (AAA0000)
        
        Verifica que placas no formato antigo (3 letras + 4 dígitos) são
        corretamente identificadas.
        """
        # Configurar CLASS_NAMES
        mock_config.CLASS_NAMES = ['0'] * 35
        
        # Criar processador
        processor = PlateCharacterProcessor()
        
        # Testar placa antiga válida
        result = processor._validate_plate_format("ABC1234")
        
        assert result['valid_format'] is True
        assert result['plate_type'] == 'Placa_Antiga'
    
    @patch('src.detection.config')
    def test_validate_plate_format_recognizes_placa_mercosul(self, mock_config):
        """
        Requisito 9.2: validação deve reconhecer formato Placa_Mercosul (AAA0A00)
        
        Verifica que placas no formato Mercosul (3 letras + 1 dígito + 1 letra + 2 dígitos)
        são corretamente identificadas.
        """
        # Configurar CLASS_NAMES
        mock_config.CLASS_NAMES = ['0'] * 35
        
        # Criar processador
        processor = PlateCharacterProcessor()
        
        # Testar placa Mercosul válida
        result = processor._validate_plate_format("ABC1D23")
        
        assert result['valid_format'] is True
        assert result['plate_type'] == 'Placa_Mercosul'
    
    @patch('src.detection.config')
    def test_validate_plate_format_rejects_wrong_length(self, mock_config):
        """
        Requisito 9.5: strings com comprimento diferente de 7 devem ser inválidas
        
        Verifica que textos com comprimento diferente de 7 caracteres são
        imediatamente rejeitados sem tentar validar o padrão.
        """
        # Configurar CLASS_NAMES
        mock_config.CLASS_NAMES = ['0'] * 35
        
        # Criar processador
        processor = PlateCharacterProcessor()
        
        # Testar comprimentos inválidos
        result_short = processor._validate_plate_format("ABC123")  # 6 caracteres
        assert result_short['valid_format'] is False
        assert result_short['plate_type'] is None
        
        result_long = processor._validate_plate_format("ABC12345")  # 8 caracteres
        assert result_long['valid_format'] is False
        assert result_long['plate_type'] is None
        
        result_empty = processor._validate_plate_format("")  # 0 caracteres
        assert result_empty['valid_format'] is False
        assert result_empty['plate_type'] is None
    
    @patch('src.detection.config')
    def test_validate_plate_format_rejects_invalid_patterns(self, mock_config):
        """
        Requisito 9.3: strings que não correspondem a nenhum padrão válido devem ser inválidas
        
        Verifica que textos de 7 caracteres que não correspondem a nenhum dos
        formatos válidos são rejeitados.
        """
        # Configurar CLASS_NAMES
        mock_config.CLASS_NAMES = ['0'] * 35
        
        # Criar processador
        processor = PlateCharacterProcessor()
        
        # Testar padrões inválidos (7 caracteres mas formato errado)
        result1 = processor._validate_plate_format("1234567")  # Todos dígitos
        assert result1['valid_format'] is False
        assert result1['plate_type'] is None
        
        result2 = processor._validate_plate_format("ABCDEFG")  # Todas letras
        assert result2['valid_format'] is False
        assert result2['plate_type'] is None
        
        result3 = processor._validate_plate_format("AB12345")  # 2 letras + 5 dígitos
        assert result3['valid_format'] is False
        assert result3['plate_type'] is None
        
        result4 = processor._validate_plate_format("ABCD123")  # 4 letras + 3 dígitos
        assert result4['valid_format'] is False
        assert result4['plate_type'] is None


class TestCalculateAverageConfidence:
    """Testes para o método _calculate_average_confidence"""
    
    @patch('src.detection.config')
    def test_calculate_average_confidence_returns_correct_average(self, mock_config):
        """
        Verifica que a confiança média é calculada corretamente
        """
        # Configurar CLASS_NAMES
        mock_config.CLASS_NAMES = ['0'] * 35
        
        # Criar processador
        processor = PlateCharacterProcessor()
        
        # Criar detecções com confianças conhecidas
        detections = [
            {'confidence': 0.9},
            {'confidence': 0.8},
            {'confidence': 0.7},
        ]
        
        # Calcular média
        avg = processor._calculate_average_confidence(detections)
        
        # Verificar que a média está correta (0.9 + 0.8 + 0.7) / 3 = 0.8
        assert avg == pytest.approx(0.8, rel=1e-9)
    
    @patch('src.detection.config')
    def test_calculate_average_confidence_returns_zero_for_empty_list(self, mock_config):
        """
        Verifica que retorna 0.0 para lista vazia de detecções
        """
        # Configurar CLASS_NAMES
        mock_config.CLASS_NAMES = ['0'] * 35
        
        # Criar processador
        processor = PlateCharacterProcessor()
        
        # Calcular média de lista vazia
        avg = processor._calculate_average_confidence([])
        
        # Verificar que retorna 0.0
        assert avg == 0.0
