import cv2
import os
from PIL import Image

train_inputDir = 'CBIS-DDSMI/TrainROI'
train_outDir = 'ROI-CBIS-DDSMI/train-224x224'

test_inputDir = 'CBIS-DDSMI/TestROI'
test_outDir = 'ROI-CBIS-DDSMI/test-224x224'

WIDTH = 224
HEIGHT = 224

def process_image(input_dir, out_dir, mode): 
   files = os.listdir(input_dir)

   #The mask image has last id 1 to Test images 
   #it must be the cropped image not the masks
   image_last_id = -1
   if mode == "test":
      image_last_id = '2'

   for f in files:      
      if mode == "test":
         resize_image(f,input_dir,out_dir, 224, 224)
      else:
         resize_image(f,input_dir,out_dir, 224, 224)

def resize_image(image_file, input_dir, out_dir ,width, height):
   print("Resizing " + image_file)
   image = Image.open(input_dir + '/' + image_file)
   width, height = image.size
   if width < 1000 or height < 1000:

      image = image.resize((224, 224), Image.ANTIALIAS)
      print(image.size)
      if not os.path.exists(out_dir):
         os.makedirs(out_dir)

      image.save(out_dir +'/' + image_file , 'png', quality=90)

#resize test images
process_image(test_inputDir, test_outDir, "test")

#resize train images
process_image(train_inputDir, train_outDir, "train")