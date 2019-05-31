import cv2
import os
import pydicom
from imutils import paths
import shutil
import random
import pandas as pd
import numpy as np



#Calcification data
calc_inputdir = 'Calc'
base_outdir = 'PNG_ROIs/calc/'
calc_test_outdir = base_outdir + 'test/'
calc_train_outdir = base_outdir + 'train/'

calc_test_lesion_type = 'Calc-Test'
calc_train_lesion_type = 'Calc-Trai' 

#Mass data
mass_inputdir = 'Mass'
mass_base_outdir = 'PNG_ROIs/mass/'
mass_test_outdir = mass_base_outdir + 'test/'
mass_train_outdir = mass_base_outdir+ 'train/'

mass_test_lesion_type = 'Mass-Test'
mass_train_lesion_type = 'Mass-Trai'


def convert_dicom_to_png(inputdir, test_lesion_type, train_lesion_type, test_output_dir, train_output_dir):


    test_list = [ f for f in  os.listdir(inputdir)]
    counter = 1
    #For to convert all the ROI images to png

    for f in test_list:
        for i in os.listdir(inputdir + '/'+ f):
            for g in os.listdir(inputdir + '/' + f + '/' + i):
                #print(g[10:])
                for h in os.listdir(inputdir + '/' + f + '/' + i + '/' + g):
                    for j in os.listdir(inputdir + '/' + f + '/' + i + '/' + g + '/' + h):
                        for k in os.listdir(inputdir + '/' + f + '/' + i + '/' + g + '/' + h +'/' + j):
                            pass
                        # print(k)
                        # print('Converting: ' + inputdir + '/' + f + '/' + i + '/' + g + '/' + h + '/' + j + '/' + k)
                            ds = pydicom.read_file(inputdir + '/' + f + '/' + i + '/' + g + '/' + h + '/' + j + '/' + k) # read dicom image
                            img = ds.pixel_array # get image array
                            
                            if g[:9] == test_lesion_type:#'Calc-Test':
                                print(" test")
                                if(len(os.listdir(inputdir + '/' + f + '/' + i + '/' + g+ '/' + h + '/' + j)) == 2):
                                    if(counter == 1):
                                        counter+=1
                                        print(test_output_dir + g[10:] + '_' + str(counter)+ '_ROI' +'.png')
                                        cv2.imwrite(test_output_dir + g[10:] + '_' + str(counter)+ '_ROI' +'.png',img) # write png image
                                    elif(counter ==2):
                                        counter = 1
                                        cv2.imwrite(test_output_dir + g[10:] + '_' + str(counter)+ '_ROI' +'.png',img) # write png image
                                elif(len(os.listdir(inputdir + '/' + f + '/' + i + '/' + g+ '/' + h + '/' + j)) == 1):
                                    pass
                                    cv2.imwrite(test_output_dir + g[10:]+ '_ROI' + '.png',img) # write png image

                            elif g[:9] == train_lesion_type:#'Calc-Trai':
                                print("train")
                                if(len(os.listdir(inputdir + '/' + f + '/' + i + '/' + g+ '/' + h + '/' + j)) == 2):
                                    if(counter == 1):
                                        counter+=1
                                        print(train_output_dir + g[14:] + '_' + str(counter)+ '_ROI' +'.png')
                                        cv2.imwrite(train_output_dir + g[14:] + '_' + str(counter)+ '_ROI' +'.png',img) # write png image
                                    elif(counter ==2):
                                        counter = 1
                                        cv2.imwrite(train_output_dir + g[14:] + '_' + str(counter)+ '_ROI' +'.png',img) # write png image
                                elif(len(os.listdir(inputdir + '/' + f + '/' + i + '/' + g+ '/' + h + '/' + j)) == 1):
                                    cv2.imwrite(train_output_dir + g[14:]+ '_ROI' + '.png',img) # write png image

#calcifications                            
convert_dicom_to_png(calc_inputdir,calc_test_lesion_type,calc_train_lesion_type,calc_test_outdir,calc_train_outdir)

#masses
convert_dicom_to_png(mass_inputdir, mass_test_lesion_type, mass_train_lesion_type, mass_test_outdir, mass_train_outdir)

    
            #print(f + '/' + i + '/' + g)
#            if(g[:6] == '1-ROI '):
#                print(g[:6])        
#                for q in os.listdir(inputdir + '\\' + f + '\\' + i + '\\' + g):
#    #                    print(len(os.listdir(inputdir + '\\' + f + '\\' + i + '\\' + g)))
#                    print('Converting: ' + inputdir + '\\' + f + '\\' + i + '\\' + g + '\\' +q)
#                    ds = pydicom.read_file(inputdir + '\\' + f + '\\' + i + '\\' + g + '\\' +q) # read dicom image
#                    img = ds.pixel_array # get image array
#                    
#                    if(f[5:9] == 'Test'):
#    #                       print(f[10:])
#                        if(len(os.listdir(inputdir + '\\' + f + '\\' + i + '\\' + g)) == 2):
#                            if(counter == 1):
#                                counter+=1
##                                cv2.imwrite(outdirTestROI + f[10:] + '_' + str(counter)+ '_ROI' +'.png',img) # write png image
#                            elif(counter ==2):
#                                counter =1
#                                cv2.imwrite(outdirTestROI + f[10:] + '_' + str(counter)+ '_ROI' +'.png',img) # write png image
#                        elif(len(os.listdir(inputdir + '\\' + f + '\\' + i + '\\' + g)) == 1):
#                            cv2.imwrite(outdirTestROI + f[10:]+ '_ROI' + '.png',img) # write png image
#                        else:
#                            print("none")
#                    elif(f[5:9] == 'Trai'):    
#                        #print(f[14:])
#                        if(len(os.listdir(inputdir + '\\' + f + '\\' + i + '\\' + g)) == 2):
#                            if(counter == 1):
#                                counter+=1
#                                cv2.imwrite(outdirTrainROI + f[14:] + '_' + str(counter)+ '_ROI' +'.png',img) # write png image
#                            elif(counter ==2):
#                                counter =1
#                            cv2.imwrite(outdirTrainROI + f[14:] + '_' + str(counter)+ '_ROI' +'.png',img) # write png image
#                        elif(len(os.listdir(inputdir + '\\' + f + '\\' + i + '\\' + g)) == 1):
#                            cv2.imwrite(outdirTrainROI + f[14:]+ '_ROI' + '.png',img) # write png image
                        

#For to convert the full mammograms to png

#for f in test_list:   # remove "[:10]" to convert all images 
#    for i in os.listdir(inputdir + '\\'+ f):
#        for g in os.listdir(inputdir + '\\' + f + '\\' + i):
#            if(g[:6] == '1-full'):
#                #print(g[:6])
#                for q in os.listdir(inputdir + '\\' + f + '\\' + i + '\\' + g):
#                    print('Converting: ' + inputdir + '\\' + f + '\\' + i + '\\' + g + '\\' +q)
#                    ds = pydicom.read_file(inputdir + '\\' + f + '\\' + i + '\\' + g + '\\' +q) # read dicom image
#                    img = ds.pixel_array # get image array
#                   
#                    if(f[5:9] == 'Test'):
#                        print(f[10:])
#                        cv2.imwrite(outdirTest + f[10:] + '.png',img) # write png image
#                    elif(f[5:9] == 'Trai'):    
#                        print(f[14:])
#                        cv2.imwrite(outdirTrain + f[14:] + '.png',img) # write png image 


#Getting Cropped Images
#for f in test_list:   # remove "[:10]" to convert all images 
#    for i in os.listdir(inputdir + '\\'+ f):
#        for g in os.listdir(inputdir + '\\' + f + '\\' + i):
#            print(g[:6])
#            print(f + '\\' + i + '\\' + g)
#            if(g[:6] == '1-crop'):
#                print(g[:6])
#                
#                for q in os.listdir(inputdir + '\\' + f + '\\' + i + '\\' + g):
#                    print(len(os.listdir(inputdir + '\\' + f + '\\' + i + '\\' + g)))
#                    print('Converting: ' + inputdir + '\\' + f + '\\' + i + '\\' + g + '\\' +q)
#                    ds = pydicom.read_file(inputdir + '\\' + f + '\\' + i + '\\' + g + '\\' +q) # read dicom image
#                    img = ds.pixel_array # get image array
#                   
#                    if(f[5:9] == 'Test'):
# #                       print(f[10:])
#                        if(len(os.listdir(inputdir + '\\' + f + '\\' + i + '\\' + g)) == 2):
#                            if(counter == 1):
#                                counter+=1
#                                cv2.imwrite(outdirTestCropped + f[10:] + '_' + str(counter) +'.png',img) # write png image
#                            elif(counter ==2):
#                                counter =1
#                                cv2.imwrite(outdirTestCropped + f[10:] + '_' + str(counter) +'.png',img) # write png image
#                        elif(len(os.listdir(inputdir + '\\' + f + '\\' + i + '\\' + g)) == 1):
#                           
#                           cv2.imwrite(outdirTestCropped + f[10:] + '.png',img) # write png image
#                        else:
#                            print("none")
#                    elif(f[5:9] == 'Trai'):    
#    #                        print(f[14:])
#                        if(len(os.listdir(inputdir + '\\' + f + '\\' + i + '\\' + g)) == 2):
#                            if(counter == 1):
#                                counter+=1
#                                cv2.imwrite(outdirTrainCropped + f[14:] + '_' + str(counter) +'.png',img) # write png image
#                            elif(counter ==2):
#                                counter = 1
#                            cv2.imwrite(outdirTrainCropped + f[14:] + '_' + str(counter) +'.png',img) # write png image
#                        elif(len(os.listdir(inputdir + '\\' + f + '\\' + i + '\\' + g)) == 1):
#                            cv2.imwrite(outdirTrainCropped + f[14:] + '.png',img) # write png image
                            

    #print(inputdir +'/' +f)
    #patiente_ID
    #print(f[14:21])
    #ds = pydicom.read_file(inputdir +'/' +f) # read dicom image
    #img = ds.pixel_array # get image array
    #cv2.imwrite(outdir + f.replace('.dcm','.png'),img) # write png image
    