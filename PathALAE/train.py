from PathALAE.models import *
from PathALAE.optimizers import *
from PathALAE.data import *
from PathALAE.utils import *

import argparse
import os
import logging
import datetime

import tensorflow as tf


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
FILTERS = config["filters"]                            # List of features in consecutive levels
ATTENTION_LEVEL = config["attention_level"]            # Which level to add an attention block
IMG_DIM = config["img_dim"]                            # The size of the input image / target image e.g. (256, 256, 3)
Z_DIM = config["z_dim"]                                # The size of the latent dimension e.g 512
BASE_DIM = tuple(config["base_dim"])                   # Shape of base feature space e.g. (8, 8, 256)

# Training
ALPHA = config["alpha"]                                 # Learning Rate - good between 0.001-0.002
GAMMA = config["gamma"]                                 # Gradient Penalty coefficient
BETA1 = config["beta1"]                                 # Adam beta1 coefficient
BETA2 = config["beta2"]                                 # Adam beta2 coefficient
BATCH_SIZE = config["batch_size"]                       # Depends on hardware
N = config["n"]                                         # The number of images in the dataset
EPOCHS = config["epochs"]                               # The number of training epochs
EMA = config["ema"]                                     # Bool - calculate EMA of weights?
DIS = config["d"]                                        # Number of display images for training progress

# Directories
DATA_DIR = config["data_dir"]                           # Path to the data directory
RUN_DIR = config["run_dir"]                             # Path to where to store all outputs from training

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
data_gen, _ = create_data_set(data_directory=DATA_DIR,
                              img_dim=IMG_DIM,
                              batch_size=BATCH_SIZE)

# Get test data - is the same for each run!
tf.random.set_seed(1234)
test_z = tf.random.normal((DIS, Z_DIM), seed=1)
test_batch = get_test_batch(data_gen)

# Create Strategy
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Everything that creates variables should be under the strategy scope.
with strategy.scope():
    # Build subnetworks
    F = MappingResidual(z_dim=Z_DIM, n_layers=F_LAYERS)

    G = build_demod_generator(z_dim=Z_DIM,
                              image_dim=IMG_DIM,
                              base_dim=BASE_DIM,
                              filters=FILTERS,
                              attention_level=ATTENTION_LEVEL
                              )

    E = build_encoder(z_dim=Z_DIM,
                      image_dim=IMG_DIM,
                      filters=FILTERS,
                      attention_level=ATTENTION_LEVEL)

    D = DiscriminatorResidual(z_dim=Z_DIM, n_layers=D_LAYERS)

    # Combine into ALAE
    alae = ALAE(x_dim=IMG_DIM,
                z_dim=Z_DIM,
                f_model=F,
                g_model=G,
                e_model=E,
                d_model=D)

    # Optimizers
    Adam_D, Adam_G, Adam_R = create_optimizers(α=ALPHA, β1=BETA1, β2=BETA2)

    alae.compile(d_optimizer=Adam_D,
                 g_optimizer=Adam_G,
                 r_optimizer=Adam_R,
                 γ=GAMMA)

callbacks = [
            Summary(log_dir=os.path.join(LOG_DIR, f"{IMG_DIM}x{IMG_DIM}/"),
                    write_graph=False,
                    #update_freq=50,  # every n batches
                    test_z=test_z,
                    test_batch=test_batch,
                    img_dir=os.path.join(OUT_DIR, f"{IMG_DIM}x{IMG_DIM}/"),
                    n=DIS,
                    weight_dir=os.path.join(WEIGHT_DIR, f"{IMG_DIM}x{IMG_DIM}/"),
                    )
        ]

if EMA:
    callbacks.append(
        ExponentialMovingAverage(weight_dir=WEIGHT_DIR, fid_dir=FID_DIR)
    )

alae.fit(x=data_gen,
         steps_per_epoch=N // BATCH_SIZE,
         epochs=EPOCHS,
         callbacks=callbacks)

end = datetime.datetime.now()
print(f"Training Complete - total time:", end - start)



