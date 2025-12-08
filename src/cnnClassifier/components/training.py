import os
from pathlib import Path
import tensorflow as tf
from cnnClassifier import logger

class Training:
    def __init__(self, config):
        self.config = config
        self.model = None

    def get_base_model(self):
        logger.info("training: Loading updated base model...")
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)
        logger.info("training: Base model loaded successfully.")

    def train(self, callback_list):

        # -----------------------------------------
        # 🔥 COMPILE MODEL (REQUIRED FOR NEW TRAINING)
        # -----------------------------------------
        logger.info("training: Compiling model...")

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.params_learning_rate
            ),
            loss="categorical_crossentropy",    # Multi-class loss
            metrics=["accuracy"]
        )

        logger.info("training: Model compiled successfully.")

        # -----------------------------------------
        # Data Generators (MULTICLASS)
        # -----------------------------------------
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=0.2
        )

        train_data = datagen.flow_from_directory(
            directory=self.config.training_data,
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            subset="training",
            class_mode="categorical"       # 🔥 REQUIRED FOR MULTI-CLASS
        )

        valid_data = datagen.flow_from_directory(
            directory=self.config.training_data,
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            subset="validation",
            class_mode="categorical"       # 🔥 REQUIRED FOR MULTI-CLASS
        )

        logger.info(f"training: Class indices: {train_data.class_indices}")

        logger.info("training: Training Started...")

        # -----------------------------------------
        # Training Loop
        # -----------------------------------------
        self.model.fit(
            train_data,
            validation_data=valid_data,
            epochs=self.config.params_epochs,
            callbacks=callback_list
        )

        # -----------------------------------------
        # Save trained model
        # -----------------------------------------
        logger.info(f"training: Saving trained model at {self.config.trained_model_path}")
        self.model.save(self.config.trained_model_path)

        logger.info("training: Training completed successfully.")
