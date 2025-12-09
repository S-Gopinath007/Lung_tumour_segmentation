import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt

# -------------------------------
# 1️⃣ Load the trained model
# -------------------------------
model = load_model("lung_tumor_classification_model.keras")  # or .h5
print("Model Loaded Successfully")

# -------------------------------
# 2️⃣ Build the model to define inputs
# -------------------------------
# Dummy call to create the input shape (224x224 RGB)
_ = model(np.zeros((1, 224, 224, 3), dtype=np.float32))

# -------------------------------
# 3️⃣ Load and preprocess the image
# -------------------------------
IMG_PATH = r"E:\Lung Tumour Segmentation\Dataset\test\Malignant cases\Malignant case (6).jpg"
img = cv2.imread(IMG_PATH)
if img is None:
    raise ValueError(f"❌ ERROR: Image not found at {IMG_PATH}")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (224, 224))
img_array = np.expand_dims(img_resized / 255.0, axis=0)

# -------------------------------
# 4️⃣ Access last convolution layer inside VGG16 base
# -------------------------------
vgg_base = model.layers[0]  # VGG16 base
last_conv_layer = vgg_base.get_layer("block5_conv3")
print("Last Conv Layer:", last_conv_layer.name)

# -------------------------------
# 5️⃣ Build Grad-CAM model
# -------------------------------
grad_model = tf.keras.models.Model(
    inputs=model.input,
    outputs=[last_conv_layer.output, model.output]
)

# -------------------------------
# 6️⃣ Compute Grad-CAM
# -------------------------------
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    predicted_class = tf.argmax(predictions[0])
    tape.watch(conv_outputs)

grads = tape.gradient(predictions[:, predicted_class], conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
conv_outputs = conv_outputs[0].numpy()
pooled_grads = pooled_grads.numpy()

for i in range(conv_outputs.shape[-1]):
    conv_outputs[:, :, i] *= pooled_grads[i]

heatmap = np.mean(conv_outputs, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap) + 1e-10  # avoid divide by zero

# -------------------------------
# 7️⃣ Overlay heatmap on original image
# -------------------------------
heatmap_resized = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
heatm
