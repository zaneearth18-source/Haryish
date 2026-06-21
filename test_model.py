import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load trained model
model = tf.keras.models.load_model("solder_model.keras")

# Class names (IMPORTANT: must match your folder names order)
class_names = [
    "Bridging",
    "Cold_Solder",
    "Excess_Solder",
    "Good_Solder",
    "Insufficient_Solder"
]

# Load test image (CHANGE THIS PATH)
img = keras.utils.load_img(
    r"C:\Users\zanee\OneDrive\Desktop\FYP\FYP_AOI_PROJECT\FYP_AOI_Dataset\Bridging\Bridging_41.jpg",
    target_size=(224, 224)
)

# Convert image to array
img_array = keras.utils.img_to_array(img)

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# Normalize (same as training)
img_array = img_array / 255.0

# Predict
predictions = model.predict(img_array)

# Get highest probability class
predicted_class = class_names[np.argmax(predictions)]
confidence = np.max(predictions)

# Output result
print("Prediction:", predicted_class)
print("Confidence:", round(confidence * 100, 2), "%")