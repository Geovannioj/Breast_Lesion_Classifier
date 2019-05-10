import cv2
import os
import pydicom
from imutils import paths
import shutil
import random

inputdir = 'CBIS-DDSM'
outdir = 'pngImg/'
outdirTest = 'TestPng\\'
outdirTrain = 'TrainPng\\'
outdirTestROI = 'TestROI\\'
outdirTrainROI = 'TrainROI\\'
outdirTestCropped = 'TestCropped\\'
outdirTrainCropped = 'TrainCropped\\'
counter = 1
#'./'
#os.mkdir(outdir)

test_list = [ f for f in  os.listdir(inputdir)]


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


#For to convert all the ROI images to png

for f in test_list:   # remove "[:10]" to convert all images 
    for i in os.listdir(inputdir + '\\'+ f):
        for g in os.listdir(inputdir + '\\' + f + '\\' + i):
#            print(g[:6])
#            print(f + '\\' + i + '\\' + g)
            if(g[:6] == '1-ROI '):
#                print(g[:6])        
                for q in os.listdir(inputdir + '\\' + f + '\\' + i + '\\' + g):
    #                    print(len(os.listdir(inputdir + '\\' + f + '\\' + i + '\\' + g)))
                    print('Converting: ' + inputdir + '\\' + f + '\\' + i + '\\' + g + '\\' +q)
                    ds = pydicom.read_file(inputdir + '\\' + f + '\\' + i + '\\' + g + '\\' +q) # read dicom image
                    img = ds.pixel_array # get image array
                    
                    if(f[5:9] == 'Test'):
    #                       print(f[10:])
                        if(len(os.listdir(inputdir + '\\' + f + '\\' + i + '\\' + g)) == 2):
                            if(counter == 1):
                                counter+=1
                                cv2.imwrite(outdirTestROI + f[10:] + '_' + str(counter)+ '_ROI' +'.png',img) # write png image
                            elif(counter ==2):
                                counter =1
                                cv2.imwrite(outdirTestROI + f[10:] + '_' + str(counter)+ '_ROI' +'.png',img) # write png image
                        elif(len(os.listdir(inputdir + '\\' + f + '\\' + i + '\\' + g)) == 1):
                            cv2.imwrite(outdirTestROI + f[10:]+ '_ROI' + '.png',img) # write png image
                        else:
                            print("none")
                    elif(f[5:9] == 'Trai'):    
                        #print(f[14:])
                        if(len(os.listdir(inputdir + '\\' + f + '\\' + i + '\\' + g)) == 2):
                            if(counter == 1):
                                counter+=1
                                cv2.imwrite(outdirTrainROI + f[14:] + '_' + str(counter)+ '_ROI' +'.png',img) # write png image
                            elif(counter ==2):
                                counter =1
                            cv2.imwrite(outdirTrainROI + f[14:] + '_' + str(counter)+ '_ROI' +'.png',img) # write png image
                        elif(len(os.listdir(inputdir + '\\' + f + '\\' + i + '\\' + g)) == 1):
                            cv2.imwrite(outdirTrainROI + f[14:]+ '_ROI' + '.png',img) # write png image
                        

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
    