import os
import numpy as np
import pandas as pd
import tensorflow as tf
import ants
import matplotlib.pyplot as plt

CLASS_SUBFOLDERS = ['MCI/', 'AD/', 'CN/']
DB_SS_PATH = 'data/output/'

TFR_PATH = 'data/output/TFR/'

TFR_TRAIN = 'train.tfrecords'
TFR_TEST = 'test.tfrecords'
TFR_VAL = 'val.tfrecords'

train_tfrec = os.path.join(TFR_PATH, TFR_TRAIN)
test_tfrec = os.path.join(TFR_PATH, TFR_TEST)
val_tfrec = os.path.join(TFR_PATH, TFR_VAL)

LABELS = {'CN': 1, 'MCI': 0, 'AD': 0}
BINARY_LABELS = {'CN': 0, 'AD': 1}

TEST_SPLIT = 0.15
VALIDATION_SPLIT = 0.15

filenames = np.array([])

# iterate all three class folders in the db
for subf in CLASS_SUBFOLDERS:
  # using the skull stripped data
  path = DB_SS_PATH + subf
  for name in os.listdir(path):
    complete_name = os.path.join(path, name)
    if os.path.isfile(complete_name):
      filenames = np.concatenate((filenames, complete_name), axis=None)

for i in range(1000):
  np.random.shuffle(filenames)

test_margin = int(len(filenames) * TEST_SPLIT)
training_set, test_set = filenames[test_margin:], filenames[:test_margin]

validation_margin = int(len(training_set) * VALIDATION_SPLIT)
training_set, validation_set = training_set[validation_margin:], training_set[:validation_margin]


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def load_image(abs_path):
    split_path = abs_path.split('/')
    label = LABELS[split_path[-2]]
    ants_image = ants.image_read(abs_path)
    img = ants_image.numpy()

    return img, label

def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array

def create_tf_record(img_filenames, tf_rec_filename):
    writer = tf.io.TFRecordWriter(tf_rec_filename)
    for meta_data in img_filenames:
        img, label = load_image(meta_data)
        feature = {'label': _int64_feature(label),
                   'image': _bytes_feature(serialize_array(img))}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()

create_tf_record(training_set, train_tfrec)
create_tf_record(test_set, test_tfrec)
create_tf_record(validation_set, val_tfrec)
