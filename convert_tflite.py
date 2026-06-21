import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("solder_model.keras")

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

# Save model
with open("solder_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model created successfully!")