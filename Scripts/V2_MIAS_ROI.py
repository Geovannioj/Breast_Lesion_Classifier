import cv2
import numpy as np 
import pydicom
import pandas as pd
from os import path
from PIL import Image

image_path = '../MIAS_New_download'
csv_data_path = 'MIAS_data_csv.csv'
benign_output_path = 'MIAS_2_ROI/0'
malignant_output_path = 'MIAS_2_ROI/1'

csv_data = pd.read_csv(csv_data_path, sep =";")
# files = os.listdir(image_path)

rs = 128
roi_cnt = 0


csv_data_indexed = csv_data.set_index('File')

def clip(v, minv=0, maxv=1024):
    v = minv if v < minv else v
    v = maxv if v > maxv else v
    return v

for ref, info in csv_data_indexed.iterrows():
    # print(info)
    try:
        xc = int(info['x'])
        yc = int(info['x'])
        r = int(info['radius'])
    except ValueError:
        continue
    img = cv2.imread(path.join(image_path, ref[:6] + '.pgm'), -1)

    if r <= rs:
        x = xc - rs
        x = clip(x)
        y = 1024 - yc - rs
        y = clip(y)
        x2 = x + rs * 2 
        y2 = y + rs * 2

        # roi = img.crop((x,y2,x2,y))        
        roi = img[y:y+rs*2, x:x+rs*2]
        patch = np.zeros((rs*2, rs*2))
        patch[0:roi.shape[0], 0:roi.shape[1]] = roi
        fn = "%s_roi%04d.png" % (ref[:6], roi_cnt)
        if info['Severity'] == 'B':
            cv2.imwrite(path.join(benign_output_path, fn), patch, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            roi_cnt += 1
        elif info['Severity'] == 'M':
            cv2.imwrite(path.join(malignant_output_path, fn), patch, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            roi_cnt += 1
    else:
        # When the ROI radius is larger than 128, 
        # extract all 9 patches that are centered 
        # around the ROI center.
        xss = [-r, -rs, r-2*rs]
        yss = [-r, -rs, r-2*rs]
        for xs in xss:
            for ys in yss:
                x = xc + xs
                x = clip(x)
                y = 1024 - yc + ys
                y = clip(y)
                x2 = x + rs * 2 
                y2 = y + rs * 2
                
                # roi = img.crop((x,y2,x2,y))
                roi = img[y:y+rs*2, x:x+rs*2]
                patch = np.zeros((rs*2, rs*2))
                patch[0:roi.shape[0], 0:roi.shape[1]] = roi
                fn = "%s_roi%04d.png" % (ref[:6], roi_cnt)
                
                if info['Severity'] == 'B':
                    cv2.imwrite(path.join(benign_output_path, fn), patch, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                    roi_cnt += 1
                elif info['Severity'] == 'M':
                    cv2.imwrite(path.join(malignant_output_path, fn), patch, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                    roi_cnt += 1
                # cv2.imwrite(path.join(outdir, fn), patch, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                # roi_cnt += 1
