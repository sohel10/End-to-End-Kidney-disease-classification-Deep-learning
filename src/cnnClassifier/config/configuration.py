import os
from pathlib import Path

# IMPORTANT: Set absolute paths so Jupyter/VSCode does not break
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CONFIG_FILE_PATH = PROJECT_ROOT / "config" / "config.yaml"
PARAMS_FILE_PATH = PROJECT_ROOT / "params.yaml"

from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
    PrepareCallbacksConfig,
    TrainingConfig,
    EvaluationConfig,
)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH,
    ):
        # Read YAML configs
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        # Create root artifacts directory
        create_directories([self.config.artifacts_root])

    # ---------------------------------------------------------
    # 1. DATA INGESTION CONFIG
    # ---------------------------------------------------------
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        cfg = self.config.data_ingestion
        create_directories([cfg.root_dir])

        return DataIngestionConfig(
            root_dir=Path(cfg.root_dir),
            source_URL=cfg.source_URL,
            local_data_file=Path(cfg.local_data_file),
            unzip_dir=Path(cfg.unzip_dir),
        )

    # ---------------------------------------------------------
    # 2. PREPARE BASE MODEL CONFIG
    # ---------------------------------------------------------
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        cfg = self.config.prepare_base_model
        create_directories([cfg.root_dir])

        return PrepareBaseModelConfig(
            root_dir=Path(cfg.root_dir),
            base_model_path=Path(cfg.base_model_path),
            updated_base_model_path=Path(cfg.updated_base_model_path),

            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES,
        )

    # ---------------------------------------------------------
    # 3. PREPARE CALLBACKS CONFIG
    # ---------------------------------------------------------
    def get_prepare_callback_config(self) -> PrepareCallbacksConfig:
        cfg = self.config.prepare_callbacks

        create_directories([
            Path(cfg.tensorboard_root_log_dir),
            Path(cfg.checkpoint_model_filepath).parent,
        ])

        return PrepareCallbacksConfig(
            root_dir=Path(cfg.root_dir),
            tensorboard_root_log_dir=Path(cfg.tensorboard_root_log_dir),
            checkpoint_model_filepath=Path(cfg.checkpoint_model_filepath)
        )

    # ---------------------------------------------------------
    # 4. TRAINING CONFIG
    # ---------------------------------------------------------
    def get_training_config(self) -> TrainingConfig:
        cfg = self.config.training
        params = self.params

        return TrainingConfig(
            root_dir=Path(cfg.root_dir),
            trained_model_path=Path(cfg.trained_model_path),
            updated_base_model_path=Path(self.config.prepare_base_model.updated_base_model_path),

            training_data=Path(cfg.training_data),

            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_learning_rate=params.LEARNING_RATE,
            params_image_size=params.IMAGE_SIZE,
            params_classes=params.CLASSES
        )

    # ---------------------------------------------------------
    # 5. EVALUATION CONFIG
    # ---------------------------------------------------------
    def get_evaluation_config(self) -> EvaluationConfig:
        cfg = self.config.evaluation

        return EvaluationConfig(
            path_of_model=Path(cfg.path_of_model),
            training_data=Path(self.config.data_ingestion.unzip_dir),
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
        )
