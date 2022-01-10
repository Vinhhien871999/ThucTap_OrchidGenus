import os
import cv2
import random
import numpy as np
from tensorflow import keras
img_size_224p = 128

path_train = '../orchid-genus-1/train'
path_test = '../orchid-genus-1/test'
categories = ['cattleya', 'dendrobium', 'oncidium', 'phalaenopsis', 'vanda']

def create_data_img(folder_path):
    imageData = []
    for category in categories:
        path = os.path.join(folder_path, category)
        class_num = categories.index(category)  # Take the Label as the Index
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))
            img_convert = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            img_resize = cv2.resize(img_convert, (img_size_224p, img_size_224p))
            imageData.append([img_resize, class_num])

    return imageData


dataTrain = create_data_img(path_train)
dataTest = create_data_img(path_test)

random.seed(10)
random.shuffle(dataTrain)

# X for Features & y for Labels
X_train, y_train, X_test, y_test = [], [], [], []

for features, label in dataTrain:
    X_train.append(features)
    y_train.append(label)

for features, label in dataTest:
    X_test.append(features)
    y_test.append(label)

# -1 in reshape, means to let Numpy define the appropriate data dimensions
X_train = np.array(X_train).reshape(-1, img_size_224p, img_size_224p, 3)
y_train = np.asarray(y_train)
X_test  = np.array(X_test).reshape(-1, img_size_224p, img_size_224p, 3)
y_test  = np.asarray(y_test)

print("X_train :", X_train.shape)
print("y_train :", y_train.shape)
print("X_test  :", X_test.shape)
print("y_test  :", y_test.shape)

print("Array of X_train :\n\n", X_train[0]) # Take the first data for example
print("\nArray of X_test  :\n\n", X_test[0])

def prep_pixels(train, test):
    # Convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # Normalize (feature scaling) to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # Return normalized images
    return train_norm, test_norm

X_train_norm, X_test_norm = prep_pixels(X_train, X_test)

print("\nArray of X_train_norm :\n\n", X_train_norm[0])
print("\nArray of X_test_norm  :\n\n", X_test_norm[0])

from tensorflow.keras.utils import to_categorical
print("Array of y_train :", y_train)
print("Array of y_test  :", y_test)

# One Hot Encode target values
y_train_encode = to_categorical(y_train)
y_test_encode  = to_categorical(y_test)

print("\nArray of y_train_encode :\n\n", y_train_encode)
print("\nArray of y_test_encode :\n\n", y_test_encode)

# import matplotlib.pyplot as plt
#
#
# nrows = 5 # ⚠️Can be Customized⚠️
# ncols = 5 # ⚠️Can be Customized⚠️
# hspace = 0
# wspace = 0
# fig, ax = plt.subplots(nrows, ncols, figsize=(10, 10))
# fig.subplots_adjust(hspace, wspace)
#
# for i in range(nrows):
#     for j in range(ncols):
#         temp = i*ncols+j                # Index looping
#         ax[i,j].imshow(X_train[temp])   # Show Features/images
#         if y_train[temp] == 0:
#             judul = "cattleya"
#         elif y_train[temp] == 1:
#             judul = "dendrobium"
#         elif y_train[temp] == 2:
#             judul = "oncidium"
#         elif y_train[temp] == 3:
#             judul = "phalaenopsis"
#         elif y_train[temp] == 4:
#             judul = "vanda"
#         ax[i,j].set_title(judul)        # Show Labels
#         ax[i,j].axis('off')             # Hide axis
# plt.show()

import gc     # Gabage Collector for cleaning deleted data from memory

del dataTrain
del dataTest
del X_train
del X_test
#del y_train  # Used later for Confusion Matrix
#del y_test   # Used later for Confusion Matrix

gc.collect()

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import plot_model


conv_base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size_224p, img_size_224p, 3))
conv_base.trainable = False
conv_base.summary()
# plot_model(conv_base, to_file='model.png', show_shapes=True, show_layer_names=False, rankdir='TB', expand_nested=False, dpi=80)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam


def define_model_mobilenetv2():
    model = Sequential()
    model.add(conv_base)  # The Feature Extractor uses a Pre-trained Model
    model.add(GlobalAveragePooling2D())
    model.add(
        Dense(5, activation='softmax'))  # This means that in the Hidden Layer there are 5 Neurons (5 Orchid Labels)
    # activation='softmax'is used because of the Multi-Class Classification problem

    # Compile Model
    opt = Adam(lr=0.0001)  # ⚠️Can be Customized⚠️
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])  # categorical_crossentropy is used because
    return model  # of the Multi-Class Classification problem


# Clean the Previous Model (retraining needs)
if "model" in globals():  # Check if the Model Variables exist
    del model
    gc.collect()

model = define_model_mobilenetv2()
model.summary()
# plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False, rankdir='TB', expand_nested=False,
#            dpi=80)

# %%time

import pandas as pd

epochs = 10       # ⚠️Can be Customized⚠️
batch_size = 64

model = define_model_mobilenetv2() # Define Model: Using MobileNetV2 which has been modified before
history = model.fit(X_train_norm, y_train_encode, epochs=epochs, batch_size=batch_size, verbose=1) # Fit model

model.save("model_without_kfold.h5")
model_csv = pd.DataFrame(history.history)
csv_file = "model_without_kfold.csv"
with open(csv_file, mode="w") as f:
  model_csv.to_csv(f)