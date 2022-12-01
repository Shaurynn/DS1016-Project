import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import SimpleITK as sitk


from dltk.io import preprocessing
from skimage import filters

from nipype.interfaces import fsl
from nipype.interfaces.fsl import BET
from nipype.testing import example_data

import tensorflow as tf
import os
import gzip
import tarfile
import shutil

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.models import load_model
#from keras.models import load_model
import tensorflow as tf
from tensorflow.python.lib.io import file_io

from google.cloud import storage

import numpy as np
# Set numpy to print only 2 decimal digits for neatness
np.set_printoptions(precision=2, suppress=True)

import h5py
import gcsfs
PROJECT_NAME = 'projectalzheimers'
CREDENTIALS = 'projectalzheimers-aee5389e9476.json'
MODEL_PATH = 'gs://bucketalzheimers/inception_model1.h5'
BUCKET_NAME = 'bucketalzheimers'


IMG_SHAPE = (78, 110, 86)
IMG_2D_SHAPE = (IMG_SHAPE[1] * 4, IMG_SHAPE[2] * 4)
#SHUFFLE_BUFFER = 5 #Subject to change
N_CLASSES = 3

import shutil



def resample_img(itk_image, out_spacing=[2.0, 2.0, 2.0]):
    ''' This function resamples images to 2-mm isotropic voxels.

        Parameters:
            itk_image -- Image in simpleitk format, not a numpy array
            out_spacing -- Space representation of each voxel

        Returns:
            Resulting image in simpleitk format, not a numpy array
    '''

    # Resample images to 2mm spacing with SimpleITK
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

def registrate(sitk_fixed, sitk_moving, bspline=False):
    ''' Perform image registration using SimpleElastix.
        By default, uses affine transformation.

        Parameters:
            sitk_fixed -- Reference atlas (sitk .nii)
            sitk_moving -- Image to be registrated
                           (sitk .nii)
            bspline -- Whether or not to perform non-rigid
                       registration. Note: it usually deforms
                       the images and increases execution times
    '''

    elastixImageFilter = sitk.ElastixImageFilter()#sitk.ElastixImageFilter()   SimpleElastix()
    elastixImageFilter.SetFixedImage(sitk_fixed)
    elastixImageFilter.SetMovingImage(sitk_moving)

    parameterMapVector = sitk.VectorOfParameterMap()
    parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
    if bspline:
        parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
    elastixImageFilter.SetParameterMap(parameterMapVector)

    elastixImageFilter.Execute()
    return elastixImageFilter.GetResultImage()

def skull_strip_nii(original_img, destination_img, frac=0.2): #
    ''' Practice skull stripping on the given image, and save
        the result to a new .nii image.
        Uses FSL-BET
        (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET/UserGuide#Main_bet2_options:)

        Parameters:
            original_img -- Original nii image
            destination_img -- The new skull-stripped image
            frac -- Fractional intensity threshold for BET
    '''

    btr = fsl.BET()
    btr.inputs.in_file = original_img
    btr.inputs.frac = frac
    btr.inputs.out_file = destination_img
    btr.cmdline
    res = btr.run()
    return res

def gz_extract(zipfile):
    file_name = (os.path.basename(zipfile)).rsplit('.',1)[0] #get file name for file within
    with gzip.open(zipfile,"rb") as f_in, open(f"{zipfile.split('/')[0]}/{file_name}","wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(zipfile) # delete zipped file

def slices_matrix_2D(img):
  ''' Transform a 3D MRI image into a 2D image, by obtaining 9 slices
      and placing them in a 4x4 two-dimensional grid.

      All 16 cuts are from a horizontal/axial view. They are selected
      from the 30th to the 60th level of the original 3D image.

      Parameters:
        img -- np.ndarray with the 3D image

      Returns:
        np.ndarray -- The resulting 2D image
  '''

  # create the final 2D image
  image_2D = np.empty(IMG_2D_SHAPE)

  # set the limits and the step
  TOP = 60
  BOTTOM = 30
  STEP = 2
  N_CUTS = 16

  # iterator for the cuts
  cut_it = TOP
  # iterator for the rows of the 2D final image
  row_it = 0
  # iterator for the columns of the 2D final image
  col_it = 0

  for cutting_time in range(N_CUTS):

    # cut
    cut = img[cut_it, :, :]
    cut_it -= STEP

    # reset the row iterator and move the
    # col iterator when needed
    if cutting_time in [4, 8, 12]:
      row_it = 0
      col_it += cut.shape[1]

    # copy the cut to the 2D image
    for i in range(cut.shape[0]):
      for j in range(cut.shape[1]):
        image_2D[i + row_it, j + col_it] = cut[i, j]
    row_it += cut.shape[0]

  # return the final 2D image, with 3 channels
  # this is necessary for working with most pre-trained nets
  return np.repeat(image_2D[None, ...], 3, axis=0).T

def load_image_2D(abs_path): #, labels
  ''' Load an image (.nii) and its label, from its absolute path.
      Transform it into a 2D image, by obtaining 16 slices and placing them
      in a 4x4 two-dimensional grid.

      Parameters:
        abs_path -- Absolute path, filename included
        labels -- Label mapper

      Returns:
        img -- The .nii image, converted into a numpy array
        label -- The label of the image (from argument 'labels')

  '''

  # obtain the label from the path (it is the last directory name)
  #label = labels[abs_path.split('/')[-2]]

  # load the image with SimpleITK
  sitk_image = sitk.ReadImage(abs_path)

  # transform into a numpy array
  img = sitk.GetArrayFromImage(sitk_image)

  # apply whitening
  img = preprocessing.whitening(img)

  # make the 2D image
  img = slices_matrix_2D(img)

  return img

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a floast_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array


def write_tfrecords(x, y, filename):
    writer = tf.io.TFRecordWriter(filename)

    for image, label in zip(x, y):
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'image': _bytes_feature(serialize_array(image)),
                'label': _int64_feature(label)
            }))
        writer.write(example.SerializeToString())

def _parse_image_function(example_proto):
    image_feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    features = tf.io.parse_single_example(example_proto, image_feature_description)
    image = tf.io.parse_tensor(features['image'], out_type=tf.double)
    image = tf.reshape(image, [344, 440, 3])

    label = tf.cast(features['label'], tf.int32)
    label = tf.one_hot(features['label'], 3)

    return image, label

def read_dataset(epochs, batch_size, filename):

    # filenames = [os.path.join(channel, channel_name + '.tfrecords')]
    dataset = tf.data.TFRecordDataset(filename)

    dataset = dataset.map(_parse_image_function, num_parallel_calls=10)
    dataset = dataset.prefetch(batch_size)                      ##4
    dataset = dataset.repeat(epochs)                            ##2
    dataset = dataset.shuffle(buffer_size=10 * batch_size)      ##1
    dataset = dataset.batch(batch_size, drop_remainder=True)    ##3


    return dataset

def model_loading():
    FS = gcsfs.GCSFileSystem(project=PROJECT_NAME,
                         token=CREDENTIALS)
    with FS.open(MODEL_PATH, 'rb') as model_file:
        model_gcs = h5py.File(model_file, 'r')
        savedModel = load_model(model_gcs)
    # print(savedModel.summary())
    print("model loading successfully completed")

    return savedModel


def preprocess(image, atlas):
    sitk_image = sitk.ReadImage(image)
    res_image = resample_img(sitk_image)
    print("image resampling successfully completed")
    atlas_img = sitk.ReadImage(atlas)
    atlas_img = resample_img(atlas_img)
    print("atlas image resampling successfully completed")

    res_array = sitk.GetArrayFromImage(res_image)
    res_array = preprocessing.resize_image_with_crop_or_pad(res_array, img_size=(128, 192, 192), mode='symmetric')
    res_array = preprocessing.whitening(res_array)

    registrated_image = registrate(atlas_img, res_image, bspline=False)
    sitk.WriteImage(registrated_image, f"./{image.split('/')[-1]}_registrated.nii")
    print("Registration successfully completed")

    registrated_image = sitk.ReadImage(f"./{image.split('/')[-1]}_registrated.nii")
    registrated_array = sitk.GetArrayFromImage(registrated_image)

    skull_strip_nii(f"./{image.split('/')[-1]}_registrated.nii", f"./{image.split('/')[-1]}_stripped.nii", frac=0.2)
    print("Skull Stripping successfully completed")
    gz_extract(f"./{image.split('/')[-1]}_stripped.nii.gz")

    image_2d = load_image_2D(f"./{image.split('/')[-1]}_stripped.nii")
 #   print(image_2d.shape)
    np.save(f"./{image.split('/')[-1]}_2d", image_2d)
    print("Image 2D conversion successfully completed")


def predict():
    image_test_array = []
    label_test_array = []


    for filename in os.listdir('./'):
        if filename.endswith('.npy'):
            image_test_array.append(np.load(f"./{filename}"))
            label_test_array.append(0)#if 'CN' in folder else 1 if 'MCI' in folder else 2)

    image_test_array = np.array(image_test_array)
    write_tfrecords(image_test_array, label_test_array, "./test.tfrecords")
    Test = read_dataset(10, 1, './test.tfrecords')
    # print(Test)
    Test_array = list(Test.take(1).as_numpy_iterator())
    print(Test_array[0][0])
    savedModel = load_model('./inception_model1.h5')
    print(savedModel.summary())
    prediction = savedModel.predict(Test_array[0][0])
    print(prediction)

    return int(prediction[0].argmax())
##############STREAMLIT PART#######################
# Add a title and intro text
st.title('Cherish Your Memories Lifelong (CYML)')
st.text("This is a web app that detects alzheimer's from mri scan")
# Create file uploader object
upload_file = st.file_uploader('Upload your scan in nifti format')

if upload_file is not None:
    with open("./input.nii", "wb") as buffer:
        shutil.copyfileobj(upload_file, buffer)
    preprocess('./input.nii', './ref_atlas.nii')

    prediction = predict()
    prediction_text = 'Normal' if prediction==0 else 'Mild Alzheimers' if prediction==1 else 'Alzheimers'
    st.write(prediction_text)
