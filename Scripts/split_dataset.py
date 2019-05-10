import pandas as pd 
import os 
from PIL import Image

#test CSVs
calc_test_csv = pd.read_csv('csv_data/calc_case_description_test_set.csv')
mass_test_csv = pd.read_csv('csv_data/mass_case_description_test_set.csv')

#train_CSVs
calc_train_csv = pd.read_csv('csv_data/calc_case_description_train_set.csv')
mass_train_csv = pd.read_csv('csv_data/mass_case_description_train_set.csv')

#test input
test_inputDir = 'test-224x224'

#train input
train_inputDir = 'train-224x224'

#test folders
test_benign_outDir = 'PatchDataset/test-224x224/0'
test_malignant_outdir = 'PatchDataset/test-224x224/1'

#train folders
train_benign_outDir = 'PatchDataset/train-224x224/0'
train_malignant_outdir = 'PatchDataset/train-224x224/1'

def split_image_label(inputDir, benign_outDir, malignant_outdir, csv_data):
    files = os.listdir(inputDir)
    for f in files:
        patient_id = f[:7]
        spliter = f[8:].split('_', 2)

        for i in range(csv_data.shape[0]):
            if csv_data.iloc[i].patient_id == patient_id and csv_data.iloc[i]['left or right breast'] == spliter[0] and csv_data.iloc[i]['image view'] == spliter[1]:
                print(f"imagem {i} de {csv_data.shape[0]}")
                print(f)
                image = Image.open(inputDir + '/' + f)
                if not os.path.exists(benign_outDir):
                    os.makedirs(benign_outDir)
                if not os.path.exists(malignant_outdir):
                    os.makedirs(malignant_outdir)                

                #split benign and malignant images
                if csv_data.iloc[i]['pathology'][:6] == 'BENIGN':
                    image.save(benign_outDir +'/' + f , 'png', quality=90)
                    
                elif csv_data.iloc[i]['pathology'][:6] == 'MALIGN':
                    image.save(malignant_outdir +'/' + f , 'png', quality=90)
                    
                else:
                    pass

#Test Images
#label mass test images
split_image_label(test_inputDir,test_benign_outDir,test_malignant_outdir, mass_test_csv)

#label calcification test images
split_image_label(test_inputDir,test_benign_outDir,test_malignant_outdir, calc_test_csv)


#Train images
split_image_label(train_inputDir, train_benign_outDir, train_malignant_outdir, mass_train_csv)

split_image_label(train_inputDir, train_benign_outDir, train_malignant_outdir, calc_train_csv)

#files = os.listdir(inputDir)
#for f in files:
#    patient_id = f[:7]
#    spliter = f[8:].split('_',2) 

#train Data
#    for i in range (calc_train_csv.shape[0]):
        
#        if calc_train_csv.iloc[i].patient_id == patient_id and calc_train_csv.iloc[i]['left or right breast'] == spliter[0] and calc_train_csv.iloc[i]['image view'] == spliter[1]:
#            print(f"imagem {i} de {calc_train_csv.shape[0]}")
#            print(f)
#            image = Image.open(inputDir + '/' + f)

            #split benign and malignant images
#            if calc_train_csv.iloc[i]['pathology'][:6] == 'BENIGN':
#                image.save(benign_outDir +'/' + f , 'png', quality=90)
#            elif calc_train_csv.iloc[i]['pathology'][:6] == 'MALIGN':
#                image.save(malignant_outdir +'/' + f , 'png', quality=90)
#            else:
#                pass

        #Mass data
#    for i in range (mass_train_csv.shape[0]):
            
#            if mass_train_csv.iloc[i].patient_id == patient_id and mass_train_csv.iloc[i]['left or right breast'] == spliter[0] and mass_train_csv.iloc[i]['image view'] == spliter[1]:
#                print(f"imagem {i} de {mass_train_csv.shape[0]}")
#                print(f)
#                image = Image.open(inputDir + '/' + f)
#                #split benign and malignant images
#                if mass_train_csv.iloc[i]['pathology'][:6] == 'BENIGN':
#                    image.save(benign_outDir +'/' + f , 'png', quality=90)
#                elif mass_train_csv.iloc[i]['pathology'][:6] == 'MALIGN':
#                    image.save(malignant_outdir +'/' + f , 'png', quality=90)
#                else:
#                    pass            

#Test data
    #Calcification data
#    for i in range (calc_test_csv.shape[0]):
        
#        if calc_test_csv.iloc[i].patient_id == patient_id and calc_test_csv.iloc[i]['left or right breast'] == spliter[0] and calc_test_csv.iloc[i]['image view'] == spliter[1]:
            
#           image = Image.open(inputDir + '/' + f)
            #split benign and malignant images
#            if calc_test_csv.iloc[i]['pathology'][:6] == 'BENIGN':
#                image.save(benign_outDir +'/' + f , 'png', quality=90)
#            elif calc_test_csv.iloc[i]['pathology'][:6] == 'MALIGN':
#                image.save(malignant_outdir +'/' + f , 'png', quality=90)
#            else:
#                pass

    #Mass data
#    for i in range (mass_test_csv.shape[0]):
            
#            if mass_test_csv.iloc[i].patient_id == patient_id and mass_test_csv.iloc[i]['left or right breast'] == spliter[0] and mass_test_csv.iloc[i]['image view'] == spliter[1]:
                
#                image = Image.open(inputDir + '/' + f)
                #split benign and malignant images
#                if mass_test_csv.iloc[i]['pathology'][:6] == 'BENIGN':
#                    image.save(benign_outDir +'/' + f , 'png', quality=90)
#                elif mass_test_csv.iloc[i]['pathology'][:6] == 'MALIGN':
#                    image.save(malignant_outdir +'/' + f , 'png', quality=90)
#                else:
#                    pass            
