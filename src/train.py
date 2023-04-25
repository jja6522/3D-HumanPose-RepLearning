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
lambda_v = 1000 # Regularizer between all predicted poses and last/next
beta = 0.1

def set_seed(seed):
    """Set seed for (1) tensorflow, (2) random and (3) numpy for stable results"""
    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def loss_function_ae(x, y, y_rec):

    # MSE between the ground truth future and predicted poses
    batch_mse = tf.reduce_sum(tf.pow(y_rec - y, 2), axis=[1, 2])
    mean_mse = tf.reduce_mean(batch_mse)

    # MSE between last pose and next predicted pose
    last_gt_pose = x[:, -1, :]
    first_pred_pose = y_rec[:, 0, :]

    batch_mse_v = tf.reduce_sum(tf.pow(last_gt_pose - first_pred_pose, 2), axis=[1])
    mean_mse_v = tf.reduce_mean(batch_mse_v)

    total_loss = mean_mse + lambda_v * mean_mse_v

    return total_loss, mean_mse, mean_mse_v


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="AE", help="AE, VAE, DLow")
    parser.add_argument("--num_epochs", type=int, default=50, help="Numer of epochs for training")
    parser.add_argument("--samples_per_epoch", type=int, default=5000, help="samples_per_epoch")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lrate", type=int, default=1.e-3, help="Learning rate")
    args = parser.parse_args()

    # Set the random sets for reproducibility
    set_seed(RANDOM_SEED)

    #######################################
    # Dataset loading
    #######################################
    train_ds = DatasetH36M('train', t_his, t_pred, actions='all')
    test_ds = DatasetH36M('test', t_his, t_pred, actions='all')

    # Define the available models
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
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lrate)
    total_loss_tracker = tf.keras.metrics.Mean()
    mse_loss_tracker = tf.keras.metrics.Mean()
    mse_v_loss_tracker = tf.keras.metrics.Mean()

    for epoch in tqdm(range(0, args.num_epochs)):
        start_time = time.time()
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
                z = model.encode(x, y)
                y_rec = model.decode(z)
                total_loss, mean_mse, mean_mse_v = loss_function_ae(x, y, y_rec)

            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            total_loss_tracker.update_state(total_loss)
            mse_loss_tracker.update_state(mean_mse)
            mse_v_loss_tracker.update_state(mean_mse_v)

        # Compute the losses at the end of each epoch
        elapsed_time = time.time() - start_time
        tqdm.write("====> Epoch %i(%.2fs): Loss: %g\tMSE: %g\tMSE_v: %g" % 
                    (epoch, elapsed_time,
                     total_loss_tracker.result(), mse_loss_tracker.result(), mse_v_loss_tracker.result()))

    # Save the model to disk
    model.save_model(args.num_epochs)

