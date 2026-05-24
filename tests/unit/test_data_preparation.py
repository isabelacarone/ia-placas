"""
Testes unitários para o módulo data_preparation.

Valida comportamento da classe DataPreparator incluindo:
- extract_datasets (Requisitos 3.1, 3.2, 3.3, 3.4)
- _fix_directory_structure (Requisito 3.2)
- validate_dataset_structure (Requisitos 3.5, 3.6)
- create_data_yaml (Requisito 3.7)
"""

import pytest
import zipfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from src.data_preparation import DataPreparator
from src.config import config


class TestExtractDatasets:
    """Testes para extract_datasets (Requisitos 3.1, 3.3, 3.4)"""
    
    def test_extract_datasets_raises_file_not_found_for_nonexistent_directory(self, tmp_path):
        """
        Requisito 3.4: deve lançar FileNotFoundError quando diretório não existe
        
        A exceção deve conter o caminho do diretório inválido.
        """
        nonexistent_dir = tmp_path / "nonexistent_directory"
        preparator = DataPreparator(tmp_path / "dados")
        
        with pytest.raises(FileNotFoundError) as exc_info:
            preparator.extract_datasets(nonexistent_dir)
        
        # Verificar que a mensagem contém o caminho
        assert str(nonexistent_dir) in str(exc_info.value)
    
    def test_extract_datasets_logs_warning_for_missing_zip(self, tmp_path, caplog):
        """
        Requisito 3.3: deve registrar WARNING quando ZIP está ausente e continuar
        
        Não deve lançar exceção, apenas registrar aviso no log.
        """
        zips_dir = tmp_path / "zips"
        zips_dir.mkdir()
        data_dir = tmp_path / "dados"
        
        # Criar apenas Treino.zip, deixar Validacao.zip e Teste.zip ausentes
        treino_zip = zips_dir / "Treino.zip"
        with zipfile.ZipFile(treino_zip, 'w') as zf:
            zf.writestr("images/test.jpg", "fake image data")
        
        preparator = DataPreparator(data_dir)
        
        # Deve executar sem lançar exceção
        preparator.extract_datasets(zips_dir)
        
        # Verificar que avisos foram registrados para ZIPs ausentes
        assert "Validacao.zip" in caplog.text or "não encontrado" in caplog.text
        assert "Teste.zip" in caplog.text or "não encontrado" in caplog.text
    
    def test_extract_datasets_extracts_all_zips_when_present(self, tmp_path):
        """
        Requisito 3.1: deve extrair todos os ZIPs para subdiretórios corretos
        
        Treino.zip -> treino/, Validacao.zip -> validacao/, Teste.zip -> teste/
        """
        zips_dir = tmp_path / "zips"
        zips_dir.mkdir()
        data_dir = tmp_path / "dados"
        
        # Criar os três arquivos ZIP
        for zip_name, content_name in [
            ("Treino.zip", "treino_file.txt"),
            ("Validacao.zip", "validacao_file.txt"),
            ("Teste.zip", "teste_file.txt")
        ]:
            zip_path = zips_dir / zip_name
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr(f"images/{content_name}", f"content of {content_name}")
        
        preparator = DataPreparator(data_dir)
        preparator.extract_datasets(zips_dir)
        
        # Verificar que os diretórios foram criados
        assert (data_dir / "treino").exists()
        assert (data_dir / "validacao").exists()
        assert (data_dir / "teste").exists()
        
        # Verificar que os arquivos foram extraídos
        assert (data_dir / "treino" / "images" / "treino_file.txt").exists()
        assert (data_dir / "validacao" / "images" / "validacao_file.txt").exists()
        assert (data_dir / "teste" / "images" / "teste_file.txt").exists()
    
    def test_extract_datasets_creates_target_directories(self, tmp_path):
        """
        Requisito 3.1: deve criar subdiretórios se não existirem
        
        Os diretórios treino/, validacao/, teste/ devem ser criados automaticamente.
        """
        zips_dir = tmp_path / "zips"
        zips_dir.mkdir()
        data_dir = tmp_path / "dados"
        
        # Criar um ZIP
        treino_zip = zips_dir / "Treino.zip"
        with zipfile.ZipFile(treino_zip, 'w') as zf:
            zf.writestr("test.txt", "test content")
        
        preparator = DataPreparator(data_dir)
        
        # Verificar que diretórios não existem antes
        assert not (data_dir / "treino").exists()
        
        preparator.extract_datasets(zips_dir)
        
        # Verificar que diretório foi criado
        assert (data_dir / "treino").exists()


class TestFixDirectoryStructure:
    """Testes para _fix_directory_structure (Requisito 3.2)"""
    
    def test_fix_directory_structure_renames_imagens_to_images(self, tmp_path, caplog):
        """
        Requisito 3.2: deve renomear 'imagens/' para 'images/' quando 'images/' não existe
        
        Deve registrar a renomeação no log com nível INFO.
        """
        target_dir = tmp_path / "treino"
        target_dir.mkdir()
        
        # Criar diretório 'imagens'
        imagens_dir = target_dir / "imagens"
        imagens_dir.mkdir()
        (imagens_dir / "test.jpg").write_text("fake image")
        
        preparator = DataPreparator(tmp_path / "dados")
        preparator._fix_directory_structure(target_dir)
        
        # Verificar que 'imagens' foi renomeado para 'images'
        assert not (target_dir / "imagens").exists()
        assert (target_dir / "images").exists()
        assert (target_dir / "images" / "test.jpg").exists()
        
        # Verificar que foi registrado no log
        assert "Renomeado" in caplog.text or "images" in caplog.text
    
    def test_fix_directory_structure_does_not_rename_when_images_exists(self, tmp_path):
        """
        Requisito 3.2: não deve renomear se 'images/' já existe
        
        Se 'images/' já existe, 'imagens/' deve permanecer intocado.
        """
        target_dir = tmp_path / "treino"
        target_dir.mkdir()
        
        # Criar ambos os diretórios
        imagens_dir = target_dir / "imagens"
        imagens_dir.mkdir()
        (imagens_dir / "old.jpg").write_text("old image")
        
        images_dir = target_dir / "images"
        images_dir.mkdir()
        (images_dir / "new.jpg").write_text("new image")
        
        preparator = DataPreparator(tmp_path / "dados")
        preparator._fix_directory_structure(target_dir)
        
        # Verificar que ambos os diretórios ainda existem
        assert (target_dir / "imagens").exists()
        assert (target_dir / "images").exists()
        
        # Verificar que os arquivos não foram movidos
        assert (target_dir / "imagens" / "old.jpg").exists()
        assert (target_dir / "images" / "new.jpg").exists()
    
    def test_fix_directory_structure_does_nothing_when_no_imagens_dir(self, tmp_path):
        """
        Caso de borda: não deve fazer nada se 'imagens/' não existe
        """
        target_dir = tmp_path / "treino"
        target_dir.mkdir()
        
        preparator = DataPreparator(tmp_path / "dados")
        
        # Não deve lançar exceção
        preparator._fix_directory_structure(target_dir)
        
        # Diretório deve permanecer vazio
        assert not (target_dir / "imagens").exists()
        assert not (target_dir / "images").exists()


class TestValidateDatasetStructure:
    """Testes para validate_dataset_structure (Requisitos 3.5, 3.6)"""
    
    def test_validate_dataset_structure_returns_correct_keys(self, tmp_path):
        """
        Requisito 3.5: deve retornar dicionário com chaves corretas
        
        Deve conter 'treino', 'validacao', 'teste', cada um com
        'images', 'labels', 'missing_labels', 'missing_images'.
        """
        data_dir = tmp_path / "dados"
        
        # Criar estrutura mínima
        for split in ["treino", "validacao", "teste"]:
            split_dir = data_dir / split
            split_dir.mkdir(parents=True)
        
        preparator = DataPreparator(data_dir)
        result = preparator.validate_dataset_structure()
        
        # Verificar chaves principais
        assert "treino" in result
        assert "validacao" in result
        assert "teste" in result
        
        # Verificar subchaves para cada split
        for split_name in ["treino", "validacao", "teste"]:
            assert "images" in result[split_name]
            assert "labels" in result[split_name]
            assert "missing_labels" in result[split_name]
            assert "missing_images" in result[split_name]
            
            # Verificar tipos
            assert isinstance(result[split_name]["images"], int)
            assert isinstance(result[split_name]["labels"], int)
            assert isinstance(result[split_name]["missing_labels"], int)
            assert isinstance(result[split_name]["missing_images"], int)
    
    def test_validate_dataset_structure_counts_images_and_labels(self, tmp_path):
        """
        Requisito 3.5: deve contar corretamente imagens e labels
        """
        data_dir = tmp_path / "dados"
        treino_dir = data_dir / "treino" / "images"
        treino_dir.mkdir(parents=True)
        
        labels_dir = data_dir / "treino" / "labels"
        labels_dir.mkdir(parents=True)
        
        # Criar 3 imagens e 2 labels
        (treino_dir / "img1.jpg").write_text("fake")
        (treino_dir / "img2.png").write_text("fake")
        (treino_dir / "img3.jpeg").write_text("fake")
        
        (labels_dir / "img1.txt").write_text("0 0.5 0.5 0.1 0.1")
        (labels_dir / "img2.txt").write_text("1 0.3 0.4 0.2 0.2")
        
        preparator = DataPreparator(data_dir)
        result = preparator.validate_dataset_structure()
        
        # Verificar contagens
        assert result["treino"]["images"] == 3
        assert result["treino"]["labels"] == 2
        assert result["treino"]["missing_labels"] == 1  # img3 sem label
        assert result["treino"]["missing_images"] == 0
    
    def test_validate_dataset_structure_detects_missing_labels(self, tmp_path, caplog):
        """
        Requisito 3.6: deve detectar imagens sem labels e registrar aviso
        """
        data_dir = tmp_path / "dados"
        treino_dir = data_dir / "treino" / "images"
        treino_dir.mkdir(parents=True)
        
        # Criar imagens sem labels correspondentes
        (treino_dir / "img1.jpg").write_text("fake")
        (treino_dir / "img2.jpg").write_text("fake")
        
        preparator = DataPreparator(data_dir)
        result = preparator.validate_dataset_structure()
        
        # Verificar que detectou imagens sem labels
        assert result["treino"]["missing_labels"] == 2
        
        # Verificar que registrou aviso no log
        assert "sem labels" in caplog.text or "missing" in caplog.text.lower()
    
    def test_validate_dataset_structure_detects_missing_images(self, tmp_path, caplog):
        """
        Requisito 3.6: deve detectar labels sem imagens e registrar aviso
        """
        data_dir = tmp_path / "dados"
        labels_dir = data_dir / "validacao" / "labels"
        labels_dir.mkdir(parents=True)
        
        # Criar labels sem imagens correspondentes
        (labels_dir / "label1.txt").write_text("0 0.5 0.5 0.1 0.1")
        (labels_dir / "label2.txt").write_text("1 0.3 0.4 0.2 0.2")
        
        preparator = DataPreparator(data_dir)
        result = preparator.validate_dataset_structure()
        
        # Verificar que detectou labels sem imagens
        assert result["validacao"]["missing_images"] == 2
        
        # Verificar que registrou aviso no log
        assert "sem imagens" in caplog.text or "missing" in caplog.text.lower()
    
    def test_validate_dataset_structure_handles_multiple_image_extensions(self, tmp_path):
        """
        Caso de borda: deve contar imagens com diferentes extensões
        
        Deve suportar .jpg, .jpeg, .png, .bmp (maiúsculas e minúsculas).
        """
        data_dir = tmp_path / "dados"
        teste_dir = data_dir / "teste" / "images"
        teste_dir.mkdir(parents=True)
        
        # Criar imagens com diferentes extensões
        (teste_dir / "img1.jpg").write_text("fake")
        (teste_dir / "img2.JPG").write_text("fake")
        (teste_dir / "img3.jpeg").write_text("fake")
        (teste_dir / "img4.JPEG").write_text("fake")
        (teste_dir / "img5.png").write_text("fake")
        (teste_dir / "img6.PNG").write_text("fake")
        (teste_dir / "img7.bmp").write_text("fake")
        (teste_dir / "img8.BMP").write_text("fake")
        
        preparator = DataPreparator(data_dir)
        result = preparator.validate_dataset_structure()
        
        # Deve contar todas as 8 imagens
        assert result["teste"]["images"] == 8


class TestCreateDataYaml:
    """Testes para create_data_yaml (Requisito 3.7)"""
    
    def test_create_data_yaml_generates_file_with_required_keys(self, tmp_path):
        """
        Requisito 3.7: deve gerar arquivo com chaves obrigatórias
        
        Deve conter: path, train, val, test, nc=35, names com 35 classes.
        """
        data_dir = tmp_path / "dados"
        output_path = tmp_path / "test_data.yaml"
        
        preparator = DataPreparator(data_dir)
        result_path = preparator.create_data_yaml(output_path)
        
        # Verificar que arquivo foi criado
        assert result_path.exists()
        assert result_path == output_path
        
        # Ler conteúdo do arquivo
        content = output_path.read_text(encoding='utf-8')
        
        # Verificar chaves obrigatórias
        assert "path:" in content
        assert "train:" in content
        assert "val:" in content
        assert "test:" in content
        assert "nc: 35" in content or "nc:35" in content
        assert "names:" in content
    
    def test_create_data_yaml_contains_correct_nc_value(self, tmp_path):
        """
        Requisito 3.7: nc deve ser exatamente 35
        """
        data_dir = tmp_path / "dados"
        output_path = tmp_path / "test_data.yaml"
        
        preparator = DataPreparator(data_dir)
        preparator.create_data_yaml(output_path)
        
        content = output_path.read_text(encoding='utf-8')
        
        # Verificar que nc é 35
        assert "nc: 35" in content or "nc:35" in content
    
    def test_create_data_yaml_contains_all_35_classes(self, tmp_path):
        """
        Requisito 3.7: names deve conter todas as 35 classes
        
        Deve incluir dígitos 0-9 e letras A-Z (exceto O).
        """
        data_dir = tmp_path / "dados"
        output_path = tmp_path / "test_data.yaml"
        
        preparator = DataPreparator(data_dir)
        preparator.create_data_yaml(output_path)
        
        content = output_path.read_text(encoding='utf-8')
        
        # Verificar que contém os 35 nomes de classes do config
        # O formato é uma lista Python, então deve aparecer como string
        assert str(config.CLASS_NAMES) in content
        
        # Verificar alguns exemplos específicos
        assert "'0'" in content
        assert "'9'" in content
        assert "'A'" in content
        assert "'Z'" in content
        # Verificar que 'O' não está presente (excluído das classes)
        # Note: 'O' pode aparecer em outras palavras, então verificamos o contexto
        class_names_str = str(config.CLASS_NAMES)
        assert "'O'" not in class_names_str
    
    def test_create_data_yaml_uses_default_path_when_none(self, tmp_path):
        """
        Requisito 3.7: deve usar config.DATA_YAML quando output_path é None
        """
        data_dir = tmp_path / "dados"
        
        preparator = DataPreparator(data_dir)
        
        with patch.object(config, 'DATA_YAML', tmp_path / "default_data.yaml"):
            result_path = preparator.create_data_yaml(output_path=None)
            
            # Deve usar o caminho padrão do config
            assert result_path == tmp_path / "default_data.yaml"
            assert result_path.exists()
    
    def test_create_data_yaml_creates_parent_directories(self, tmp_path):
        """
        Caso de borda: deve criar diretórios pais se não existirem
        """
        data_dir = tmp_path / "dados"
        output_path = tmp_path / "nested" / "dir" / "data.yaml"
        
        # Verificar que diretórios não existem
        assert not output_path.parent.exists()
        
        preparator = DataPreparator(data_dir)
        result_path = preparator.create_data_yaml(output_path)
        
        # Verificar que diretórios foram criados
        assert output_path.parent.exists()
        assert result_path.exists()
    
    def test_create_data_yaml_contains_correct_split_paths(self, tmp_path):
        """
        Requisito 3.7: deve conter caminhos corretos para splits
        
        train: treino, val: validacao, test: teste
        """
        data_dir = tmp_path / "dados"
        output_path = tmp_path / "test_data.yaml"
        
        preparator = DataPreparator(data_dir)
        preparator.create_data_yaml(output_path)
        
        content = output_path.read_text(encoding='utf-8')
        
        # Verificar caminhos dos splits
        assert "train: treino" in content or "train:treino" in content
        assert "val: validacao" in content or "val:validacao" in content
        assert "test: teste" in content or "test:teste" in content


class TestDataPreparatorInitialization:
    """Testes para inicialização do DataPreparator"""
    
    def test_init_uses_config_data_dir_when_none(self):
        """
        Deve usar config.DATA_DIR quando data_dir não é fornecido
        """
        preparator = DataPreparator()
        
        assert preparator.data_dir == config.DATA_DIR
    
    def test_init_uses_provided_data_dir(self, tmp_path):
        """
        Deve usar data_dir fornecido quando especificado
        """
        custom_dir = tmp_path / "custom_data"
        preparator = DataPreparator(custom_dir)
        
        assert preparator.data_dir == custom_dir
    
    def test_init_creates_zip_mapping(self, tmp_path):
        """
        Deve criar mapeamento correto de ZIPs para diretórios
        """
        data_dir = tmp_path / "dados"
        preparator = DataPreparator(data_dir)
        
        # Verificar mapeamento
        assert "Treino.zip" in preparator.zip_mapping
        assert "Validacao.zip" in preparator.zip_mapping
        assert "Teste.zip" in preparator.zip_mapping
        
        assert preparator.zip_mapping["Treino.zip"] == data_dir / "treino"
        assert preparator.zip_mapping["Validacao.zip"] == data_dir / "validacao"
        assert preparator.zip_mapping["Teste.zip"] == data_dir / "teste"
