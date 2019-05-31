import cv2
import os
import scipy
from skimage.filters import gaussian
import numpy as np
from PIL import Image


#input
input_dir = '../_PatchDatasetNormalized'

#outputs
test_benign_output = '../_Fourier_and_GCN/test-224x224/0'
test_malignant_output = '../_Fourier_and_GCN/test-224x224/1'

train_benign_output = '../_Fourier_and_GCN/train-224x224/0'
train_malignant_output = '../_Fourier_and_GCN/train-224x224/1'

validation_benign_output = '../_Fourier_and_GCN/validation-224x224/0'
validation_malignant_output = '../_Fourier_and_GCN/validation-224x224/1'

folders = os.listdir(input_dir)

def process_image(kinds,benign_output, malignant_output):
    for kind in kinds:
        files = os.listdir(input_dir + '/' + folder + '/'+ kind)
        for file in files:
            image = cv2.imread(input_dir + '/' + folder + '/'+ kind + '/'+ file, cv2.IMREAD_GRAYSCALE)
            # filtered_img = cv2.blur(image,(5,5))
            f = np.fft.fft2(image)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20*np.log(np.abs(fshift))
            magnitude_spectrum = np.asarray(magnitude_spectrum, dtype=np.uint8)

            #read image
            #apply gaussiam filter
            if kind == '0':
                print(file)
                #save in the benign output
                cv2.imwrite(benign_output + '/' + file, magnitude_spectrum)
            elif kind == '1':
                #save in the malignant ouput
                print(file)
                cv2.imwrite(malignant_output + '/' + file, magnitude_spectrum)

# img_path = input_dir + "/test-224x224/0/P_00032_RIGHT_CC_1_2_ROI.png"
# img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
# f = np.fft.fft2(img)
# fshift = np.fft.fftshift(f)
# magnitude_spectrum = 20*np.log(np.abs(fshift))
# magnitude_spectrum = np.asarray(magnitude_spectrum, dtype=np.uint8)

# cv2.imwrite("Transformed_P_00032_RIGHT_CC_1_2_ROI.png", magnitude_spectrum)

# cv2.imshow("Image", magnitude_spectrum)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# fourier = np.fft.fftshift(fourier)
# fourier = abs(fourier)
# fourier = np.log10(fourier)
# lowest = np.nanmin(fourier[np.isfinite(fourier)])
# highest = np.nanmax(fourier[np.isfinite(fourier)])
# original_range = highest - lowest
# norm_fourier = (fourier - lowest) / original_range
# norm_fourier_img = Image.fromarray(norm_fourier)

# norm_fourier_img.convert("L").save("Transformder_P_00032_RIGHT_CC_1_2_ROI.png")


for folder in folders:

    if folder == "test-224x224":
        print(folder)
        kinds = os.listdir(input_dir + '/' + folder)
        process_image(kinds, test_benign_output, test_malignant_output)     
    elif folder == "train-224x224":
        print(folder)
        kinds = os.listdir(input_dir + '/' + folder)
        process_image(kinds, train_benign_output,train_malignant_output)
        
    elif folder == "validation-224x224":
        print(folder)
        process_image(kinds,validation_benign_output, validation_malignant_output)
