from keras.applications import MobileNet
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
import os
import numpy as np
from os import listdir
from os.path import isfile, join

# MobileNet was designed to work on 224 x 224 pixel input images sizes
img_rows, img_cols = 224, 224

input_shape = (img_rows, img_cols, 3)

# Re-loads the MobileNet model without the top or FC layers
MobileNet = MobileNet(weights = 'imagenet', 
                 include_top = False,
                     input_shape = (img_rows, img_cols, 3))

# Here we freeze the last 4 layers 
# Layers are set to trainable as True by default
for layer in MobileNet.layers:
    layer.trainable = False
    
    
# Let's print our layers 
for (i,layer) in enumerate(MobileNet.layers):
    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)
    
print(MobileNet.summary())

def lw(bottom_model, num_classes):
    """creates the top or head of the model that will be 
    placed ontop of the bottom layers"""
    
    top_model = bottom_model.output

#     top_model = GlobalAveragePooling2D()(top_model)
#     top_model = Dense(1024,activation='relu')(top_model)
#     top_model = Dense(128, activation='relu')(top_model)
#     top_model = Dense(num_classes,activation='softmax')(top_model)
    top_model = Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape)(top_model)
    top_model = (Conv2D(64, (3, 3), activation='relu'))(top_model)
    top_model = (MaxPooling2D(pool_size=(2, 2)))(top_model)
    top_model = (Dropout(0.25))(top_model)
    top_model = (Flatten())(top_model)
    top_model = (Dense(128, activation='relu'))(top_model)
    top_model = (Dropout(0.5))(top_model)
    top_model = (Dense(num_classes, activation='softmax'))(top_model)
    
    return top_model

    # Set our class number to 3 (Young, Middle, Old)
num_classes = 2

FC_Head = lw(MobileNet, num_classes)

model = Model(inputs = MobileNet.input, outputs = FC_Head)

#model.save_weights("face.h5")

print(model.summary())

#MobileNet.layers.output

#model.layers

train_data_dir = 'Desktop/model1/pics/train/'
validation_data_dir = 'Desktop/model1/pics/test/'

# Let's use some data augmentaiton 
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=45,
      width_shift_range=0.3,
      height_shift_range=0.3,
      horizontal_flip=True,
      fill_mode='nearest')
 
validation_datagen = ImageDataGenerator(rescale=1./255)
 
# set our batch size (typically on most mid tier systems we'll use 16-32)
batch_size = 32
 
print("Train Data:")
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')

print("Test Data:")
 
validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')
nb_train_samples=len(train_generator.classes)
nb_validation_samples=len(validation_generator.classes)
checkpoint = ModelCheckpoint("face.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)

earlystop = EarlyStopping(monitor = 'val_loss', 
                          min_delta = 0, 
                          patience = 3,
                          verbose = 1,
                          restore_best_weights = True)

# we put our call backs into a callback list
callbacks = [earlystop, checkpoint]

# We use a very small learning rate 
model.compile(loss = 'categorical_crossentropy',
              optimizer = RMSprop(lr = 0.001),
              metrics = ['accuracy'])


epochs = 1
batch_size = 25

history = model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    callbacks = callbacks,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size)
acc=history.history['accuracy'][-1]
f = open("acc.txt", "w")
f.write(str(acc*100))
f.close()
