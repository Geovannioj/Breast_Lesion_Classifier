import cv2
import os
from PIL import Image

#Base paths
basedir = 'PNG_ROIs/'
calc_basedir = basedir + 'calc'
mass_basedir = basedir + 'mass'

#Calc input paths
calc_test_dir = calc_basedir + '/test/'
calc_train_dir = calc_basedir + '/train/'

#Calc output paths

#Mass input paths
mass_test_dir = mass_basedir + '/test/'
mass_train_dir = mass_basedir + '/train/'

#Mass output paths

WIDTH = 224
HEIGHT = 224

def process_image(input_dir, out_dir): 
   files = os.listdir(input_dir)
   for f in files:      
      if f[-3:] == "png":
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

#process_image(calc_test_dir,calc_test_dir)
process_image(calc_train_dir,calc_train_dir)
process_image(mass_test_dir, mass_test_dir)
process_image(mass_train_dir, mass_train_dir)
