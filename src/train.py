import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers

import sys
import random
import numpy as np
import matplotlib.pyplot as plt

from utils.dataset_h36m import DatasetH36M
from utils.visualization import render_animation
import time
import argparse
from tqdm import tqdm, trange

from models import AE, VAE

###########################################################
# FIXME: Hack required to enable GPU operations by TF RNN
###########################################################
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

###########################################################
# Global configurations
###########################################################
RANDOM_SEED = 7 # For luck

# Hyperparameters for training/testing
t_his = 25 # number of past motions (c)
t_pred = 100 # number of future motions (t)
nk = 5 # sample images for reconstruction

# Inference configurations
#all_algos = ['ae']



def set_seed(seed):
    """Set seed for (1) tensorflow, (2) random and (3) numpy for stable results"""
    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="AE", help="AE, VAE, DLow)")
    parser.add_argument("--num_epochs", type=int, default=30, help="Numer of epochs for training")
    parser.add_argument("--samples_per_epoch", type=int, default=5000, help="samples_per_epoch")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lrate", type=int, default=1.e-3, help="Learning rate")
    args = parser.parse_args()

    # Set the random sets for reproducibility
    set_seed(RANDOM_SEED)

    # Load the train/test splits
    train_ds = DatasetH36M('train', t_his, t_pred, actions='all')
    test_ds = DatasetH36M('test', t_his, t_pred, actions='all')

    # Load the model to train
    model_dict = {
                  "AE": AE(name='autoencoder',
                           traj_dim = train_ds.traj_dim,
                           t_his = t_his,
                           t_pred = t_pred),
                  "VAE": VAE(name='vae',
                             traj_dim = train_ds.traj_dim,
                             t_his = t_his,
                             t_pred = t_pred)
                  }

    if args.model in model_dict:
        model = model_dict[args.model]

    #######################################
    # Model training
    #######################################

    print("Training model:", type(model))
    model.summary()

    # Loss function, metrics and optimizer
    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lrate)
    training_loss_tracker = tf.keras.metrics.Mean(name="training_loss")

    train_stats = trange(0, desc='training_loss')
    for r in tqdm(range(0, args.num_epochs)):
        generator = train_ds.sampling_generator(num_samples=args.samples_per_epoch, batch_size=args.batch_size)
        for traj_np in generator:

            # Remove the center hip joint for training
            traj_np = traj_np[..., 1:, :]

            # Stack all joints
            traj_np = traj_np.reshape(traj_np.shape[0], traj_np.shape[1], -1)

            # Transpose for selecting frames instead of batches
            traj = np.ascontiguousarray(np.transpose(traj_np, (1, 0, 2)))

            # Transpose back to batches and take past and future motions for encoding
            x = np.transpose(traj[:t_his], (1, 0, 2))
            y = np.transpose(traj[t_his:], (1, 0, 2))

            with tf.GradientTape() as tape:
                x_enc = model.encode(x)
                y_rec = model.decode(x_enc)
                total_loss = loss(y, y_rec)
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            training_loss_tracker.update_state(total_loss)
            train_stats.set_description('training_loss=%g' % training_loss_tracker.result())

    # Save the model to disk
    autoencoder.encoder.save(f"models/encoder-{num_epochs}.model")
    autoencoder.decoder.save(f"models/decoder-{num_epochs}.model")

