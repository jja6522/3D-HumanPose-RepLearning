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
# NOTE: Hack required to enable GPU operations by TF RNN
###########################################################
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

###########################################################
# Global configurations
###########################################################
RANDOM_SEED = 7 # For luck

# General Hyperparameters for training/testing
t_his = 25 # number of past motions (c)
t_pred = 100 # number of future motions (x)

# AE Hyperparameters
lambda_v_ae = 1000  # AE Weighting factor for (c) between last past pose and first predicted pose

# VAE Hyperparameters
lambda_v_vae = 1000 # VAE Weighting factor for (c) between last past pose and first predicted pose
beta_vae = 0.1 # VAE Regularizer for KL Divergence; higher values aim at precision and lower values aim at diversity


def set_seed(seed):
    """Set seed for (1) tensorflow, (2) random and (3) numpy for stable results"""
    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def loss_function_ae(c, x, x_rec):

    # MSE between the ground truth future and predicted poses
    batch_mse = tf.reduce_sum(tf.pow(x_rec - x, 2), axis=[1, 2])
    mean_mse = tf.reduce_mean(batch_mse)

    # MSE between last pose and next predicted pose
    last_gt_pose = c[:, -1, :]
    first_pred_pose = x_rec[:, 0, :]

    batch_mse_v = tf.reduce_sum(tf.pow(last_gt_pose - first_pred_pose, 2), axis=[1])
    mean_mse_v = tf.reduce_mean(batch_mse_v)

    total_loss = mean_mse + lambda_v_ae * mean_mse_v

    return total_loss, mean_mse, mean_mse_v


def loss_function_vae(c, x, x_rec, mu, logvar):

    # MSE between the ground truth future and predicted poses
    batch_mse = tf.reduce_sum(tf.pow(x_rec - x, 2), axis=[1, 2])
    mean_mse = tf.reduce_mean(batch_mse)

    # MSE between last pose and next predicted pose
    last_gt_pose = c[:, -1, :]
    first_pred_pose = x_rec[:, 0, :]

    batch_mse_v = tf.reduce_sum(tf.pow(last_gt_pose - first_pred_pose, 2), axis=[1])
    mean_mse_v = tf.reduce_mean(batch_mse_v)

    # KLD for the distribution difference of mu and logvar
    batch_kld = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), axis=1)
    mean_kld = tf.reduce_mean(batch_kld)

    total_loss = mean_mse + lambda_v_vae * mean_mse_v + beta_vae * mean_kld

    return total_loss, mean_mse, mean_mse_v, mean_kld


@tf.function
def train_step_ae(model, x, c, optimizer,
                  total_loss_tracker, mse_loss_tracker, mse_v_loss_tracker):

    with tf.GradientTape() as tape:
        z = model.encode(x, c)
        x_rec = model.decode(z, c)
        total_loss, mean_mse, mean_mse_v = loss_function_ae(c, x, x_rec)

    # Backpropagation to optimize the total loss
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Update the losses
    total_loss_tracker.update_state(total_loss)
    mse_loss_tracker.update_state(mean_mse)
    mse_v_loss_tracker.update_state(mean_mse_v)


@tf.function
def train_step_vae(model, x, c, optimizer,
                   total_loss_tracker, mse_loss_tracker, mse_v_loss_tracker, kld_loss_tracker):

    with tf.GradientTape() as tape:
        mu, logvar = model.encode(x, c)
        z = model.reparameterize(mu, logvar)
        x_rec = model.decode(z, c)
        total_loss, mean_mse, mean_mse_v, mean_kld = loss_function_vae(c, x, x_rec, mu, logvar)

    # Backpropagation to optimize the total loss
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Update the losses
    total_loss_tracker.update_state(total_loss)
    mse_loss_tracker.update_state(mean_mse)
    mse_v_loss_tracker.update_state(mean_mse_v)
    kld_loss_tracker.update_state(mean_kld)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="vae", help="ae, vae, dlow")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of epochs for testing")
    parser.add_argument("--samples_per_epoch", type=int, default=5000, help="Number of samples per epoch")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lrate", type=int, default=1.e-3, help="Learning rate")
    args = parser.parse_args()

    # Set the random sets for reproducibility
    set_seed(RANDOM_SEED)

    #######################################
    # Dataset loading
    #######################################
    train_ds = DatasetH36M('train', t_his, t_pred, actions='all')

    # Define the available models
    model_dict = {
                  "ae": AE(name='ae',
                           traj_dim = train_ds.traj_dim,
                           t_his = t_his,
                           t_pred = t_pred),
                  "vae": VAE(name='vae',
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

    # Loss metric trackers for training the AE
    ae_loss_tracker = tf.keras.metrics.Mean()
    ae_mse_tracker = tf.keras.metrics.Mean()
    ae_mse_v_tracker = tf.keras.metrics.Mean()

    # Loss metric trackers for training the CVAE
    vae_loss_tracker = tf.keras.metrics.Mean()
    vae_mse_tracker = tf.keras.metrics.Mean()
    vae_mse_v_tracker = tf.keras.metrics.Mean()
    vae_kld_tracker = tf.keras.metrics.Mean()

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
            c = np.transpose(traj[:t_his], (1, 0, 2))
            x = np.transpose(traj[t_his:], (1, 0, 2))

            # Train step in a tf.function for speed
            if model.name == 'ae':
                train_step_ae(model, x, c, optimizer,
                              ae_loss_tracker,
                              ae_mse_tracker,
                              ae_mse_v_tracker)

            elif model.name == 'vae':
                train_step_vae(model, x, c, optimizer,
                               vae_loss_tracker,
                               vae_mse_tracker,
                               vae_mse_v_tracker,
                               vae_kld_tracker)

        # Compute the losses at the end of each epoch
        elapsed_time = time.time() - start_time
        if model.name == 'ae':
            tqdm.write("====> [%s] Epoch %i(%.2fs)\tLoss: %g\tMSE: %g\tMSE_v: %g" %
                        (model.name, epoch, elapsed_time,
                         ae_loss_tracker.result(), ae_mse_tracker.result(), ae_mse_v_tracker.result()))
        elif model.name == 'vae':
            tqdm.write("====> [%s] Epoch %i(%.2fs)\tLoss: %g\tMSE: %g\tMSE_v: %g\tKLD: %g" %
                        (model.name, epoch, elapsed_time,
                         vae_loss_tracker.result(), vae_mse_tracker.result(), vae_mse_v_tracker.result(), vae_kld_tracker.result()))

    # Save the model to disk
    model.save_model(args.num_epochs)

