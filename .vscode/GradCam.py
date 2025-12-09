from tensorflow.keras.models import load_model

model_path = r"E:\Lung Tumour Segmentation\lung_tumor_classification_model.h5"
model = load_model(model_path)
print("Model Loaded Successfully!")
