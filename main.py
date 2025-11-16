import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

DATA_DIR = "Data"
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32


train = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="int",
    color_mode="grayscale",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    subset="training",
    validation_split=0.2,
)

test = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="int",
    color_mode="grayscale",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    subset="validation",
    validation_split=0.2,
)

model = Sequential()

model.add(Conv2D(64, (3,3), strides = 1, padding = 'same', input_shape = (128,128,1)))
model.add(Dropout(0.05))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2), strides = 2, padding = 'valid'))

model.add(Conv2D(32, (3,3), strides = 1, padding = 'same'))
model.add(Dropout(0.15))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2), strides = 2, padding = 'valid'))

model.add(Conv2D(16, (3,3), strides = 1, padding = 'same'))
model.add(Dropout(0.05))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2), strides = 1, padding = 'valid'))

model.add(Flatten())
model.add(Dense(units = 64, activation = 'relu'))
model.add(Dropout(0.05))
model.add(BatchNormalization())

model.add(Dense(units = 128, activation = 'relu'))
model.add(Dropout(0.15))
model.add(BatchNormalization())

model.add(Dense(units = 64, activation = 'relu'))
model.add(Dropout(0.05))
model.add(BatchNormalization())

model.add(Dense(units = 2, activation = 'softmax'))

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.005) , loss = 'sparse_categorical_crossentropy' , metrics = ['accuracy'])

model.summary()



history = model.fit(train, epochs = 120,
                    validation_data = test)


