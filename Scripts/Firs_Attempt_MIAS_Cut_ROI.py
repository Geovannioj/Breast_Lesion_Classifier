import cv2
import numpy as np
import pydicom
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy
import scipy.misc
from PIL import Image
from glob import glob
import re

#Paths
image_path = '../MIAS_New_download'#'../MIASDBv1.21-PNG'
output_path = '../MIAS_DB_ROI'
csv_data_path = 'MIAS_data_csv.csv'
only_roi_path = '../MIAS_ONLY_ROI'

#csvLoad
csv_data = pd.read_csv(csv_data_path, sep =";")

files = os.listdir(image_path)

print(csv_data.head(10))

def saveImage(i, out_dir):
    print(csv_data['File'].iloc[i][:6])
    x = int(csv_data['x'].iloc[i])
    y = int(csv_data['y'].iloc[i])
    radius = int(csv_data['radius'].iloc[i])

    print(x)
    print(y)
    print("#############")
    x1 = int(x - radius)
    y1 = 1024 - int(y - radius)
    x2 = int(x + radius)
    y2 = 1024 - int(y + radius)
    print(x1)
    print(x2)
    print(y1)
    print(y2)

    # image = cv2.imread(image_path + "/" + csv_data['File'].iloc[i] + ".png",cv2.IMREAD_UNCHANGED)
    immage = Image.open(image_path + "/" + csv_data['File'].iloc[i][:6] + ".pgm")

    # im = immage.crop((x2,y2,x1,y1))
    im = immage.crop((x1,y2,x2,y1))
    im.save(out_dir + "/" + csv_data['File'].iloc[i][:6] + ".png")
    

    #PLOT IMAGE WITH ROI
    im_arr = np.asarray(immage)
    # convert rgb array to opencv's bgr format
    im_arr_bgr = cv2.cvtColor(im_arr, cv2.COLOR_RGB2BGR)
    img_001_rect = cv2.rectangle(im_arr_bgr, (x1, y1), (x2, y2), (255,255,255), 2)
    fig, ax = plt.subplots()
    fig.set_size_inches([6, 6])
    ax.imshow(img_001_rect.data, cmap='gray')
    plt.show()

    return im

# saveImage(213)

def global_contrast_normalization(image, s, lmda, epsilon,out_dir):
    X = image

    X_average = np.mean(X)
    print("Mean:", X_average)
    X = X - X_average

    contrast = np.sqrt(lmda+np.mean(X**2))
    X = s * X / max(contrast, epsilon)

    scipy.misc.imsave(out_dir, X)



for i in range(0, csv_data['File'].size): 
    print(csv_data['File'].iloc[i][:6])
    if csv_data['Severity'].iloc[i] == 'M':
        cropped_img = saveImage(i, '../MIAS_DB_ROI/Normalized/1')
        # global_contrast_normalization(cropped_img,1,10,0.000000001,'../MIAS_DB_ROI/Normalized/1/' + csv_data['File'].iloc[i]+ "malignant" + "_GCN_ROI.png")
    elif csv_data['Severity'].iloc[i] == 'B':
        cropped_img = saveImage(i, '../MIAS_DB_ROI/Normalized/0')
        # global_contrast_normalization(cropped_img,1,10,0.000000001,'../MIAS_DB_ROI/Normalized/0/' + csv_data['File'].iloc[i]+ "benign" + "_GCN_ROI.png")



