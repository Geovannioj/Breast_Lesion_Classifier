import pandas as pd  
import numpy as np   
import os  
import shutil

#Base paths
basedir = 'PNG_ROIs/'
calc_basedir = basedir + 'calc'
mass_basedir = basedir + 'mass'

#Calc input paths
calc_test_dir = calc_basedir + '/test/'
calc_train_dir = calc_basedir + '/train/'

#Calc output paths
#calc_output_base = 'PNG_ROIs/calc'
calc_output_test_dir = calc_basedir + '/test/Masks/'
calc_output_train_dir = calc_basedir + '/train/Masks/'

#Mass input paths
mass_test_dir = mass_basedir + '/test/'
mass_train_dir = mass_basedir + '/train/'

#Mass output paths
#mass_output_dir = mass_basedir + 'mass/'
mass_output_test_dir = mass_basedir + '/test/Masks/'
mass_output_train_dir = mass_basedir + '/train/Masks/'

mask_default_number = '2_ROI.png'

def split_masks(inputdir, outputdir):
    files = os.listdir(inputdir)
    for file in files: 
        if file[-9:] == mask_default_number:
            print("##########################################")
            print(" #########  Moving File : ################")
            print(inputdir  + file)
            print(" TO: ")
            print(outputdir + file)
            print("##########################################")
            shutil.move(inputdir + file, outputdir + file)
            


split_masks(calc_test_dir, calc_output_test_dir)
split_masks(mass_test_dir, mass_output_test_dir)
split_masks(calc_train_dir, calc_output_train_dir)
split_masks(mass_train_dir, mass_output_train_dir)