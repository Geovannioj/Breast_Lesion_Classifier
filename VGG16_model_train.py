#imports
# set the matplotlib backend so figures can be saved in the background
import matplotlib
import joblib
import configuration
from cancernet import CancerNet
matplotlib.use("Agg") 
import sys
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping,R    educeLROnPlateau, Callback
from keras.optimizers import Adam,Adagrad, SGD
from keras.utils import np_utils
from keras.applications import VGG19
from keras.applications import VGG16
from keras import optimizers
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras import backend as k
import h5py
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
import glob
from keras.regularizers import l2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="New_VGG16_plot.png",
    help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

#properties

# initialize our number of epochs, initial learning rate, and batch
# size
NUM_EPOCHS = 200 
INIT_LR = 1e-4
BS = 32

height = 224 
width = 224
train_to_path = 'PatchDatasetNormalized/train-224x224'
test_to_path = 'PatchDatasetNormalized/validation-224x224'
validation_to_path = 'PatchDatasetNormalized/test-224x224'
# determine the total number of image paths in training, validation,
# and testing directories
trainPaths = list(paths.list_images(train_to_path))
totalTrain = len(trainPaths)
totalVal = len(list(paths.list_images(validation_to_path)))
totalTest = len(list(paths.list_images(test_to_path)))

# account for skew in the labeled data
trainLabels = [int(p.split(os.path.sep)[-2]) for p in trainPaths]
trainLabels = np_utils.to_categorical(trainLabels)
classTotals = trainLabels.sum(axis=0)
classWeight = classTotals.max() / classTotals

#image Generators

# initialize the training training data augmentation object
trainAug = ImageDataGenerator(
    #rescale=1 / 255.0,
    rotation_range=40,
    zoom_range=[0.5,1.5],
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.5,
    horizontal_flip=True,
    #vertical_flip=True,
    fill_mode="nearest")

# initialize the validation (and testing) data augmentation object
valAug = ImageDataGenerator()#rescale=1 / 255.0)

# initialize the training generator
trainGen = trainAug.flow_from_directory(
    train_to_path,
    class_mode="categorical",
    target_size=(height, width),
    color_mode="rgb", #change to gray_scale
    shuffle=True,
    batch_size=BS)

# initialize the validation generator
valGen = valAug.flow_from_directory(
    validation_to_path,
    class_mode="categorical",
    target_size=(height, width),
    color_mode="rgb",
    shuffle=False,
    batch_size=BS)
# initialize the testing generator
testGen = valAug.flow_from_directory(
    test_to_path,
    class_mode="categorical",
    target_size=(height, width),
    color_mode="rgb",
    shuffle=False,
    batch_size=BS)


#model import
# initialize VGG16 model
model = VGG16(weights='imagenet', include_top=False, input_shape=(width,height,3))

#free the base model layers
for layer in model.layers:
    layer.trainable = False

x = model.output 
x = GlobalMaxPooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(2, activation='softmax',kernel_regularizer=l2(1e-5))(x)

model_final = Model(input= model.input, output= predictions)

#compile the model for the last layers
model_final.compile(optimizer='Adam', loss= 'categorical_crossentropy', metrics=['accuracy'])

#train the new layers for a few epochs
hist = model_final.fit_generator(trainGen,
        steps_per_epoch=totalTrain //BS,
        validation_data=valGen,
        nb_val_samples=totalVal //BS,
        epochs=3,
        class_weight = { 0:1.0, 1:1. },
        verbose=1)


#Callbackcs options

#reduceelr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=5, verbose=1)
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_loss', save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_loss', min_delta =0, patience=10, verbose=1, mode='auto',)
callbacks = [early, checkpoint]

#freeze layers 
frozen_layers = 10
for layer in model_final.layers[:frozen_layers]:
    layer.trainable = False

for layer in model_final.layers[frozen_layers:]:
    layer.trainable = True


#compile

model_final.summary()

optm2 = SGD(lr=INIT_LR, decay= 1e-6, momentum=0.9, nesterov=True)
model_final.compile(loss = "categorical_crossentropy", optimizer= optm2, metrics=["accuracy"])

#train

hist = model_final.fit_generator(
    trainGen,
    steps_per_epoch=totalTrain // BS,
    validation_data=valGen,
    nb_val_samples=totalVal // BS,
    epochs=NUM_EPOCHS,
    class_weight= { 0:1.0, 1:1. },
    callbacks=callbacks)

#test

# reset the testing generator and then use our trained model to
# make predictions on the data
print("[INFO] evaluating network...")
testGen.reset()

predIdxs = model_final.predict_generator(testGen,
    steps=(totalTest // BS + 1))

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
print(len(testGen.classes))
print(len(predIdxs))

predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testGen.classes, predIdxs,
    target_names=testGen.class_indices.keys()))

# compute the confusion matrix and and use it to derive the raw
# accuracy, sensitivity, and specificity
cm = confusion_matrix(testGen.classes, predIdxs)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

# show the confusion matrix, accuracy, sensitivity, and specificity
print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))

#save the model
dump(model_final, 'VGG16_Model.pkl')