import numpy
import scipy
import scipy.misc
from PIL import Image
import os

train_input_dir = 'PatchDataset/train-224x224'
train_out_dir = 'PatchDatasetNormalized/train-224x224'

test_input_dir = 'PatchDataset/test-224x224'
test_out_dir = 'PatchDatasetNormalized/test-224x224'

val_input_dir = 'PatchDataset/validation-224x224'
val_out_dir = 'PatchDatasetNormalized/validation-224x224'


def global_contrast_normalization(filename, s, lmda, epsilon,out_dir, out_name):
    X = numpy.array(Image.open(filename))

    X_average = numpy.mean(X)
    print("Mean:", X_average)
    X = X - X_average

    contrast = numpy.sqrt(lmda+numpy.mean(X**2))
    X = s * X / max(contrast, epsilon)

    scipy.misc.imsave(out_dir+out_name, X)


def get_files(input_dir, out_dir):
       
    sub_dirs = os.listdir(input_dir)

    for sub_dir in sub_dirs:
        print(sub_dir)
        images = os.listdir(input_dir+'/'+sub_dir)
        output_place = out_dir + '/' + sub_dir + '/'
        for image in images:
            print(image)
            #print(output_place + image)
            global_contrast_normalization(input_dir + '/' + sub_dir + '/' + image,1,10,0.000000001,output_place, image)

#get_files(val_input_dir, val_out_dir)
#get_files(test_input_dir, test_out_dir)
get_files(train_input_dir, train_out_dir)

#nameFile = 'PatchDataset/train-224x224/0/P_00364_RIGHT_CC_1_2_ROI.png'
#global_contrast_normalization(nameFile, 1, 10, 0.000000001)