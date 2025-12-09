
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
dir=os.listdir('E:\\Lung Tumour Segmentation\\The IQ-OTHNCCD lung cancer dataset')
print(dir)

import splitfolders
splitfolders.ratio('E:\\Lung Tumour Segmentation\\The IQ-OTHNCCD lung cancer dataset', output='Dataset', seed=42, ratio=(0.8, 0.1, 0.1))

for split in ['train','test','val']:
    path=f'E:/Lung Tumour Segmentation/Dataset/{split}'
    print(f'{split.upper()} SET:')
    for folder in os.listdir(path):
         cls_path = os.path.join(path, folder)
         print(f'  {folder}: {len(os.listdir(cls_path))} images')

#Data Augumentation

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8,1.2],
    rescale=1./255
)
val_datagen=ImageDataGenerator(rescale=1./255)
test_datagen=ImageDataGenerator(rescale=1./255)

train_datagen=train_datagen.flow_from_directory(
    'E:/Lung Tumour Segmentation/Dataset/train',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)
val_datagen=val_datagen.flow_from_directory(
    'E:/Lung Tumour Segmentation/Dataset/val',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)
test_datagen=test_datagen.flow_from_directory(
    'E:/Lung Tumour Segmentation/Dataset/test',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

#Early Stopping
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
early_stopping=EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

"""
#Model Building
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense
model=Sequential([
    Conv2D(32,(3,3),strides=(1,1),activation='relu',input_shape=(224,224,3),padding='same'),
    MaxPooling2D(2,2),
    Conv2D(64,(3,3),strides=(1,1),activation='relu',input_shape=(224,224,3),padding='same'),
    MaxPooling2D(2,2),
    Conv2D(128,(3,3),strides=(1,1),activation='relu',input_shape=(224,224,3),padding='same'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128,activation='relu'),
    Dense(3,activation='softmax')
])
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(
    train_datagen,
    validation_data=val_datagen,
    epochs=20,
    callbacks=[early_stopping]
)
test_loss,test_acc=model.evaluate(test_datagen)
print("Accuracy:",test_acc)"""

#VGG 16 Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Dropout
base_model=VGG16(weights='imagenet',include_top=False,input_shape=(224,224,3))
base_model.trainable=False
model=Sequential([
    base_model,
    Flatten(),
    Dense(256,activation='relu'),
    Dropout(0.5),
    Dense(3,activation='softmax')
])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(
    train_datagen,
    validation_data=val_datagen,
    epochs=20,
    callbacks=[early_stopping],
    verbose=1
)
test_loss,test_acc=model.evaluate(test_datagen)
print("Accuracy:",test_acc)

#Save the model
model.save('lung_tumor_classification_model.h5')

#Grad Cam Visualization
"""import cv2
img_path="E:\\Lung Tumour Segmentation\\Dataset\\test\\Malignant cases\\Malignant case (6).jpg"
img=cv2.imread(img_path)
img_resized=cv2.resize(img,(224,224))
img_normalized=img_resized/255.0
input_array=np.expand_dims(img_normalized,axis=0)  

#Last conv layer
last_conv_layer=model.get_layer('conv2d_2')
grad_model=tf.keras.models.Model([model.inputs],[last_conv_layer.output,model.output])  

#Compute the gradients
with tf.GradientTape() as Tape: 
    conv_outputs,predictions=grad_model(input_array)
    predicted_class=tf.argmax(predictions[0])
    tape.watch(conv_outputs)
gradients=Tape.gradient(predictions[:,predicted_class],conv_outputs)
weights=tf.reduce_mean(gradients,axis=(0,1,2))
cam=tf.reduce_sum(tf.multiply(weights,conv_outputs),axis=-1).numpy()
cam=tf.nn.relu(cam)

heatmap=cam/tf.reduce_max(cam)
heatmap=heatmap.numpy()

heatmap=cv2.resize(heatmap,(img.shape[1],img.shape[0]))
heatmap=np.uint8(255*heatmap)
heatmap=cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)

#Superimpose the heatmap on original image
superimposed_img=cv2.addWeighted(img,0.6,heatmap,0.4,0)
cv2.imshow('Grad-CAM',superimposed_img)
cv2.waitKey(120)
cv2.destroyAllWindows()"""