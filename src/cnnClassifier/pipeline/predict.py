import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


# CLASS LABELS (update if needed)
CLASS_NAMES = ["Cyst", "Normal", "Stone", "Tumor"]


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # Correct model path
        model_path = os.path.join("artifacts", "training", "model1.h5")

        # Load the model
        model = load_model(model_path)

        # Load and preprocess image
        img = image.load_img(self.filename, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0   # normalize
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        result = np.argmax(predictions, axis=1)[0]

        # Map result to class name
        predicted_label = CLASS_NAMES[result]

        return [{"prediction": predicted_label}]
