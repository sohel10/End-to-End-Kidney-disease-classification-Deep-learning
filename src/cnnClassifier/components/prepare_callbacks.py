import os
import tensorflow as tf
from pathlib import Path
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import PrepareCallbacksConfig


class PrepareCallback:
    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config

    @property
    def _create_tb_callback(self):
        logger.info("Creating TensorBoard callback...")
        return tf.keras.callbacks.TensorBoard(
            log_dir=str(self.config.tensorboard_root_log_dir),
            histogram_freq=1
        )

    @property
    def _create_checkpoint_callback(self):
        logger.info("Creating ModelCheckpoint callback...")
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=str(self.config.checkpoint_model_filepath),
            save_best_only=True
        )

    def get_tb_ckpt_callbacks(self):
        logger.info("Returning TensorBoard + Checkpoint callbacks...")
        return [
            self._create_tb_callback,
            self._create_checkpoint_callback
        ]
