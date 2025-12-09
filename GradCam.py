import tensorflow as tf
from tensorflow.keras.models import load_model

model_path = r"E:\Lung Tumour Segmentation\lung_tumor_classification_model.h5"
model = load_model(model_path)
print("Model Loaded Successfully!")



#Loading the image for the classification
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

img_path=r"E:\Lung Tumour Segmentation\images (1).jfif"
loaded_img=load_img(img_path,target_size=(224,224))
img_array=img_to_array(loaded_img)
img_array=img_array/255.0
input_array=np.expand_dims(img_array,axis=0)

#Prediction
predictions=model.predict(input_array)
class_labels=['Bengin','Malignant','Normal']
predicted_class_index=np.argmax(predictions,axis=1)[0]
predicted_class_label=class_labels[predicted_class_index]   
print(f"Predicted Class: {predicted_class_label}")

model.summary()

#Grad Cam Visualization
vgg_model=model.layers[0]

last_conv_layer=vgg_model.get_layer('block5_conv3')
grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[last_conv_layer.output, model.output]
)


#Compute the gradient of the top predicted class for our input image
with tf.GradientTape() as tape:
    conv_outputs,predictions=grad_model(input_array)
    top_class_channel=predictions[:,predicted_class_index]  
grads=tape.gradient(top_class_channel,conv_outputs)

#pooling the gradients over all the axes
pooled_grads=tf.reduce_mean(grads,axis=(0,1,2))

#weught the feature map with the pooled gradients
conv_outputs=conv_outputs[0]    
heatmap=conv_outputs @ pooled_grads[...,tf.newaxis]
heatmap=tf.squeeze(heatmap)
heatmap=tf.maximum(heatmap,0)/tf.math.reduce_max(heatmap)
import matplotlib.pyplot as plt
plt.matshow(heatmap.numpy())

#superimpose the heatmap on original image
import cv2  
heatmap=cv2.resize(heatmap.numpy(),(loaded_img.size[0],loaded_img.size[1]))
heatmap=np.uint8(255*heatmap)
heatmap=cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
superimposed_img=heatmap*0.4 + np.array(loaded_img)
cv2.imwrite(r"E:\Lung Tumour Segmentation\grad_cam_output.jpg",superimposed_img)
import tensorflow as tf

