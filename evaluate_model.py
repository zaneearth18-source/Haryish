import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


# Load trained model
model = keras.models.load_model("solder_model.keras")


# Load dataset
test_ds = keras.utils.image_dataset_from_directory(
    "FYP_AOI_Dataset",
    image_size=(224,224),
    batch_size=16,
    shuffle=False
)

# Save class names BEFORE map()
class_names = test_ds.class_names


# Normalize images
normalization_layer = keras.layers.Rescaling(1./255)

test_ds = test_ds.map(
    lambda x,y:(normalization_layer(x),y)
)

# Evaluate overall performance

loss, accuracy = model.evaluate(test_ds)

print("--------------------------------")
print("Test Accuracy:", accuracy*100,"%")
print("Test Loss:", loss)
print("--------------------------------")


# Prediction

y_true = []
y_pred = []


for images, labels in test_ds:

    predictions = model.predict(images)

    predicted_classes = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)



# Class names


print("\nClassification Report\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)



# Confusion Matrix

cm = confusion_matrix(
    y_true,
    y_pred
)


plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names
)


plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png", bbox_inches="tight")


print("Confusion matrix saved")