import tensorflow as tf
from pathlib import Path
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    # -------------------------------------------------------
    # Load the pre-trained base model (e.g., VGG16)
    # -------------------------------------------------------
    def get_base_model(self):
        logger.info("Loading base CNN model (VGG16 / MobileNet)...")

        self.model = tf.keras.applications.VGG16(
            input_shape=self.config.params_image_size,   # [224,224,3]
            weights=self.config.params_weights,          # "imagenet"
            include_top=self.config.params_include_top   # False
        )

        logger.info("Base model loaded successfully.")
        logger.info(self.model.summary())
        return self.model

    # -------------------------------------------------------
    # Add custom classification head for your dataset
    # -------------------------------------------------------
    def update_base_model(self):
        logger.info("Freezing all layers of the base model...")

        for layer in self.model.layers:
            layer.trainable = False

        logger.info("Adding custom classification head...")

        # Flatten features
        x = tf.keras.layers.Flatten()(self.model.output)

        # Dense layer
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        # 🚨 FINAL OUTPUT LAYER (IMPORTANT)
        # Uses number of classes from params.yaml → CLASSES: 4
        output_layer = tf.keras.layers.Dense(
            self.config.params_classes,   
            activation="softmax"
        )(x)

        # Build final model
        self.updated_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=output_layer,
            name="updated_base_model"
        )

        logger.info("Updated base model created successfully.")

        # Save Updated Model
        logger.info(f"Saving updated base model to: {self.config.updated_base_model_path}")
        self.updated_model.save(self.config.updated_base_model_path)

        logger.info("Updated base model saved successfully.")
        return self.updated_model
