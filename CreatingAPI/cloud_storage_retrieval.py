from keras.models import load_model
import tensorflow as tf
from tensorflow.python.lib.io import file_io

from google.cloud import storage

import h5py
import gcsfs
PROJECT_NAME = 'projectalzheimers'
CREDENTIALS = 'projectalzheimers-aee5389e9476.json'
MODEL_PATH = 'gs://bucketalzheimers/inception_model1.h5'
BUCKET_NAME = 'bucketalzheimers'



# # def write_read(bucket_name, blob_name):
# """Write and read a blob from GCS using file-like IO"""
#     # The ID of your GCS bucket
#     # bucket_name = "your-bucket-name"

#     # The ID of your new GCS object
# blob_name = "inception_model_first_try.h5"

# storage_client = storage.Client()
# bucket = storage_client.bucket(BUCKET_NAME)
# blob = bucket.blob(blob_name)

# # Mode can be specified as wb/rb for bytes mode.
# # See: https://docs.python.org/3/library/io.html
# with blob.open("w") as f:
#     f.write('./inception_model1.h5')

    # with blob.open("r") as f:
    #     print(f.read())
