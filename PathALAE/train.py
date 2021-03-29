from PathALAE.models import *
from PathALAE.optimizers import *
from PathALAE.data import *
from PathALAE.utils import *

import argparse
import os
import logging
import datetime

C = "/home/simon/PycharmProjects/PathologyALAE/PathALAE/configs/celebA_64.yaml"

# -------------- ARG PARSE ------------#
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="The YAML config file for the run", default=C)
args = parser.parse_args()
config = ConfigParser(args.config)
# ------------------------------------- #


# -------------------------- PARAMETERS  ------------------------- #

# Model
F_LAYERS = config["F_layers"]                          # Number of layers in Mapping network (F)
D_LAYERS = config["D_layers"]                          # Number of layers in Discriminator
DECODER_TYPE = config["decoder_type"]                  # Whether AdaIN or DeMod
ENCODER_TYPE = config["encoder_type"]                  # Whether InNorm or Residual
BASE_FEATURES = config["base_features"]                # Shape of base feature space e.g. (8, 8, 256)
FILTERS = config["filters"]                            # List of features in consecutive levels
ATTENTION_LEVEL = config["attention_level"]            # Which level to add an attention block
IMG_DIM = config["img_dim"]
Z_DIM = config["z_dim"]

# Training
ALPHA = config["alpha"]
GAMMAS = config["gammas"]
BATCH_SIZE = config["batch_size"]
N = config["n"]
EPOCHS = config["epochs"]

# Directories
DATA_DIR = config["data_dir"]
RUN_DIR = config["run_dir"]

# ------------------------------------------------------------ #
# HACKS
# 6 OR 7 - 6 allows for GTX1050-ti
os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"] = str(6)
# Remove stupid warnings
tf.get_logger().setLevel(logging.ERROR)
# ------------------------------------------------------------ #

# Setup directories
LOG_DIR, OUT_DIR, WEIGHT_DIR, FID_DIR = initialise_run_directories(RUN_DIR)


# Start timer
start = datetime.datetime.now()


# Create dataset
data_gen = create_data_set(data_directory=DATA_DIR,
                           img_dim=IMG_DIM,
                           batch_size=BATCH_SIZE)

# Get test data - is the same for each run!
tf.random.set_seed(1234)
test_z = tf.random.normal((8, Z_DIM), seed=1)
test_batch = get_test_batch(data_gen)

# Create Strategy
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Everything that creates variables should be under the strategy scope.
with strategy.scope():


