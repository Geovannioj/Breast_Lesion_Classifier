import pandas as pd 
import os 
from PIL import Image
import shutil 


#test CSVs
calc_test_csv = pd.read_csv('csv_data/calc_case_description_test_set.csv')
mass_test_csv = pd.read_csv('csv_data/mass_case_description_test_set.csv')

#train_CSVs
calc_train_csv = pd.read_csv('csv_data/calc_case_description_train_set.csv')
mass_train_csv = pd.read_csv('csv_data/mass_case_description_train_set.csv')

#Base paths
basedir = 'PNG_ROIs/'
calc_basedir = basedir + 'calc'
mass_basedir = basedir + 'mass'

#Calc input paths
calc_test_dir = calc_basedir + '/test/'
calc_train_dir = calc_basedir + '/train/'

#Calc output paths
calc_output_test_benign_dir = calc_basedir + '/test/0/'
calc_output_test_malignant_dir = calc_basedir + '/test/1/'

calc_output_train_benign_dir = calc_basedir + '/train/0/'
calc_output_train_malignant_dir = calc_basedir + '/train/1/'


#Mass input paths
mass_test_dir = mass_basedir + '/test/'
mass_train_dir = mass_basedir + '/train/'

#Mass output paths
#mass_output_dir = mass_basedir + 'mass/'
mass_output_train_benign_dir = mass_basedir + '/train/0/'
mass_output_train_malignant_dir = mass_basedir + '/train/1/'

mass_output_test_benign_dir = mass_basedir + '/test/0/'
mass_output_test_malignant_dir = mass_basedir + '/test/1/'

def split_image_label(inputDir, benign_outDir, malignant_outdir, csv_data):
    files = os.listdir(inputDir)
    for f in files:
        if f[-3:] == "png":
            patient_id = f[:7]
            spliter = f[8:].split('_', 2)
            print(patient_id)
            print(spliter[0])
            print(spliter[1])
            
            for i in range(csv_data.shape[0]):
                #print("Procurando imagem local com do csv")
                #print(csv_data.iloc[i].patient_id)
                #print(csv_data.iloc[i]['left or right breast'] == spliter[0])
                #print(csv_data.iloc[i]['image view'] == spliter[1])

                if csv_data.iloc[i].patient_id == patient_id and csv_data.iloc[i]['left or right breast'] == spliter[0] and csv_data.iloc[i]['image view'] == spliter[1]:
                    print(f"imagem {i} de {csv_data.shape[0]}")
                    print(f)
                    #image = Image.open(inputDir + '/' + f)
                    if not os.path.exists(benign_outDir):
                        os.makedirs(benign_outDir)
                    if not os.path.exists(malignant_outdir):
                        os.makedirs(malignant_outdir)                

                    #split benign and malignant images
                    if csv_data.iloc[i]['pathology'][:6] == 'BENIGN':
                        try:
                            shutil.move(inputDir + f, benign_outDir + f)
                        except:
                            print("Image not found")
                    # image.save(benign_outDir +'/' + f , 'png', quality=90)
                        
                    elif csv_data.iloc[i]['pathology'][:6] == 'MALIGN':
                        #image.save(malignant_outdir +'/' + f , 'png', quality=90)
                        try:
                            shutil.move(inputDir + f, malignant_outdir + f)
                        except:
                            print("Image not found")
                        
                    else:
                        pass

#Test Images
#label calcification test images
#split_image_label(calc_test_dir,calc_output_test_benign_dir,calc_output_test_malignant_dir, calc_test_csv)

#label calcification train images
#split_image_label(calc_train_dir,calc_output_train_benign_dir,calc_output_train_malignant_dir, calc_train_csv)


#Train images
#split_image_label(mass_test_dir, mass_output_test_benign_dir, mass_output_test_malignant_dir, mass_test_csv)

split_image_label(mass_train_dir, mass_output_train_benign_dir, mass_output_train_malignant_dir, mass_train_csv)
