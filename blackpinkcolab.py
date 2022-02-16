import tensorflow as tf
import os
import zipfile
from os import path, getcwd, chdir
path = f"{getcwd()}/../content/drive/MyDrive/data.zip"
from google.colab import drive
drive.mount('/content/drive')


zip_ref = zipfile.ZipFile(path, 'r')
zip_ref.extractall("/data1234/data.zip")
zip_ref.close()



class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self,epoch,logs={}):
    if logs.get('accuracy') is not None and logs.get('accuracy')>0.96:
      self.model.stop_training=True
      print("Reached 96% accuracy so cancelling training!")
callbacks=[myCallback()]                  

jennie='/data1234/data.zip/data/jennie'
lisa='/data1234/data.zip/data/jisoo'
rose='/data1234/data.zip/data/lisa'
jisoo='/data1234/data.zip/data/rose'
    
data2='/data1234/data.zip/data'

model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(512,activation='relu'),
        tf.keras.layers.Dense(1024,activation='relu'),
        tf.keras.layers.Dense(4,activation='softmax')
    ])

from tensorflow.keras.optimizers import RMSprop 
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen =ImageDataGenerator(rotation_range=45,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(data2,target_size=(360,360),color_mode = "rgb",class_mode="categorical",shuffle=True,seed=42,batch_size=64)
print(train_generator.class_indices)
a=train_generator.class_indices
keys=a.keys()
values=a.values()
print(keys)
print(values)
history = model.fit(train_generator,epochs=25,callbacks=[myCallback()])
print(history.history['accuracy'][-1])


import numpy as np
from google.colab import files
from keras.preprocessing import image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

uploaded = files.upload()

for fn in uploaded.keys():
  path = '/content/' + fn
  img = image.load_img(path, target_size=(360, 360))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)
  main=max(classes)
  index=np.argmax(classes)
  print(main)
  print(list(classes))
  print(index)
  labels=['jennie','jisoo','lisa','rose']
  print(labels[index])

  
  