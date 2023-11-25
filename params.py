import os
import numpy as np

##################  VARIABLES  ##################
# DATA_SIZE = "1k" # ["1k", "200k", "all"]
# CHUNK_SIZE = 200
# GCP_PROJECT = "wagon-bootcamp-403414" # TO COMPLETE
# GCP_PROJECT_WAGON = "wagon-public-datasets"
# BQ_DATASET = "taxifare"
# BQ_REGION = "EU"
MODEL_TARGET = "local"
##################  CONSTANTS  #####################
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "data")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "training_outputs")
