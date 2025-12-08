import os
from pathlib import Path
import json
import tensorflow as tf
from cnnClassifier import logger


class Evaluation:
    def __init__(self, config):
        self.config = config
        self.model = None

    def load_model(self):
        logger.info(f"evaluation: Loading trained model from {self.config.path_of_model}")
        self.model = tf.keras.models.load_model(self.config.path_of_model)
        logger.info("evaluation: Model loaded successfully.")

    def evaluate(self):
        logger.info("evaluation: Preparing validation dataset...")

        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=0.2
        )

        val_data = datagen.flow_from_directory(
            directory=self.config.training_data,
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            subset="validation",
            class_mode="categorical",
            shuffle=False
        )

        logger.info("evaluation: Running evaluation on validation dataset...")

        loss, accuracy = self.model.evaluate(val_data)

        logger.info(f"evaluation: Validation Loss = {loss}")
        logger.info(f"evaluation: Validation Accuracy = {accuracy}")

        # Save results
        result = {
            "validation_loss": float(loss),
            "validation_accuracy": float(accuracy)
        }

        os.makedirs("artifacts/evaluation", exist_ok=True)
        result_path = Path("artifacts/evaluation/metrics.json")

        with open(result_path, "w") as f:
            json.dump(result, f, indent=4)

        logger.info(f"evaluation: Metrics saved at {result_path}")

        return loss, accuracy
