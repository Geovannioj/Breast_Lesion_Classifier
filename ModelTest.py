import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
import matplotlib
import configuration
matplotlib.use("Agg")

# import the necessary packages
from keras.optimizers import Adagrad
from keras.utils import np_utils
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import argparse

# # construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="plot.png",
        help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

 # initialize our number of epochs, initial learning rate, and batch
 # size
NUM_EPOCHS = 20
INIT_LR = 1e-2
BS = 1

# determine the total number of image paths in training, validation,
# and testing directories
trainPaths = list(paths.list_images(configuration.TRAIN_PATH))
totalTrain = len(trainPaths)
totalVal = len(list(paths.list_images(configuration.VAL_PATH)))
totalTest = len(list(paths.list_images(configuration.TEST_PATH)))

# account for skew in the labeled data
trainLabels = [int(p.split(os.path.sep)[-2]) for p in trainPaths]
trainLabels = np_utils.to_categorical(trainLabels)
classTotals = trainLabels.sum(axis=0)
classWeight = classTotals.max() / classTotals

# initialize the training training data augmentation object
trainAug = ImageDataGenerator(
 	rescale=1 / 255.0,
 	rotation_range=20,
 	zoom_range=0.05,
 	width_shift_range=0.1,
 	height_shift_range=0.1,
 	shear_range=0.05,
 	horizontal_flip=True,
 	vertical_flip=True,
 	fill_mode="nearest")

 # initialize the validation (and testing) data augmentation object
valAug = ImageDataGenerator(rescale=1 / 255.0)

 # initialize the training generator
trainGen = trainAug.flow_from_directory(
 	configuration.TRAIN_PATH,
 	class_mode="categorical",
 	target_size=(1500, 902),
 	color_mode="rgb",
 	shuffle=True,
 	batch_size=BS)

# initialize the validation generator
valGen = valAug.flow_from_directory(
 	configuration.VAL_PATH,
 	class_mode="categorical",
 	target_size=(1500, 902),
 	color_mode="rgb",
 	shuffle=False,
 	batch_size=BS)

# initialize the testing generator
testGen = valAug.flow_from_directory(
 	configuration.TEST_PATH,
 	class_mode="categorical",
 	target_size=(1500, 902),
 	color_mode="rgb",
 	shuffle=False,
 	batch_size=BS)


model_base = MobileNet(weights='imagenet', include_top=False)

model_layer = model_base.output
model_layer = GlobalAveragePooling2D()(model_layer)
model_layer = Dense(1024, activation='relu')(model_layer)
model_layer = Dense(512, activation='relu')(model_layer)
preds = Dense(3,activation='softmax')(model_layer)

model = Model(inputs=model_base.input, outputs = preds)

for layer in model.layers[:20]:
    layer.trainable = False
for layer in model.layers[20:]:
    layer.trainable = True

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) 
#included in our dependencies

train_generator=train_datagen.flow_from_directory(configuration.TRAIN_PATH, # this is where you specify the path to the main data folder
                                                 target_size=(1500,902),
                                                 color_mode='rgb',
                                                 batch_size=10,
                                                 class_mode='categorical',
                                                 shuffle=True)

setp_size_train=train_generator.n//train_generator.batch_size
# fit the model
#H = model.fit_generator(
#	train_generator,
#	steps_per_epoch=setp_size_train,
#	epochs=NUM_EPOCHS)
H = model.fit_generator(
            trainGen,
            steps_per_epoch=totalTrain // BS,
            validation_data=valGen,
            validation_steps=totalVal // BS,
            class_weight=classWeight,
            epochs=NUM_EPOCHS)

# reset the testing generator and then use our trained model to
# make predictions on the data
print("[INFO] evaluating network...")
testGen.reset()
predIdxs = model.predict_generator(testGen,
	steps=(totalTest // BS))

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testGen.classes, predIdxs,target_names=testGen.class_indices.keys()))

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

# plot the training loss and accuracy
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
