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
# ABSOLUTE_PATH = os.path.dirname(__file__)
LOCAL_DATA_PATH1 = os.path.join(os.path.expanduser('~'), "code", "gomezpaula658", "eye_spy_peekaboo_for_health", "data")
# LOCAL_REGISTRY_PATH =  os.path.join(ABSOLUTE_PATH)
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "data")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "training_outputs")
