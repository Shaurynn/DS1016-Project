import os
import ants
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

db = 'data/raw_data'
description = pd.read_csv('data/data.csv')

REG_Folder = 'data/registered_data/'
Output_Folder = 'data/output/'

fixed = "data/tpl-MNI305_T1w.nii.gz"
head_mask = "data/tpl-MNI305_desc-head_mask.nii.gz"
brain_mask = "data/tpl-MNI305_desc-brain_mask.nii.gz"

row_index = description.index[description['Image Data ID'] == image_ID].tolist()[0]
row = description.iloc[row_index]
label = row['Group']

labelled_reg_folder = os.path.join(REG_Folder, label, image_ID)
labelled_output_file = os.path.join(Output_Folder, label, image_ID)

def preprocess_nii(filename, path):

    REG_Folder = 'data/registered_data/'
    Output_Folder = 'data/output/'

    fixed = "data/tpl-MNI305_T1w.nii.gz"
    head_mask = "data/tpl-MNI305_desc-head_mask.nii.gz"
    brain_mask = "data/tpl-MNI305_desc-brain_mask.nii.gz"
    moving = f"{path}/{filename}"

    image_ID = filename[:-4]
    row_index = description.index[description['Image Data ID'] == image_ID].tolist()[0]
    row = description.iloc[row_index]
    label = row['Group']

    labelled_reg_folder = os.path.join(REG_Folder, label, image_ID)
    labelled_output_file = os.path.join(Output_Folder, label, image_ID)


    ! antsRegistrationSyNQuick.sh -d 3 -f {fixed} -m {moving} -o {labelled_reg_folder}_ -p f -n 4
    ! antsBrainExtraction.sh -d 3 -a {labelled_reg_folder}_InverseWarped.nii.gz -e {head_mask} -m {brain_mask} -o {labelled_output_file}_{label}_
    for f in os.listdir(os.path.join(REG_Folder, label)):
        os.remove(os.path.join(REG_Folder, label, f))

for nii in os.listdir(db):
    try:
        preprocess_nii(nii, db)
    except RuntimeError:
        print('Exception with', os.path.join(nii))
