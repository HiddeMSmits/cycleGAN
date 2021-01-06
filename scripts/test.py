from tensorflow import keras
from PIL import Image
import numpy
import os
os.chdir('/Users/Hidde/kaggle/')
#datagen = ImageDataGenerator()
#train_generator = datagen.flow_from_directory(directory= '/Users/Hidde/kaggle/monet_jpg', class_mode='binary', batch_size=64)

#model.fit_generator(train_generator, steps_per_epoch=16, validation_data=val_it, validation_steps=8)

test = Image.open('/Users/Hidde/kaggle/monet_jpg/0a5075d42a.jpg')
pix = numpy.array(test)
print(pix)