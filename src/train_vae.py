import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers

import random
import numpy as np
import matplotlib.pyplot as plt

from utils.dataset_h36m import DatasetH36M
from utils.visualization import render_animation
import time
import argparse
from tqdm import tqdm, trange

from models import VAE

###########################################################
# Tensorflow hack to enable GPU operations for RNNs
###########################################################
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

###########################################################
# Global configurations
###########################################################
RANDOM_SEED = 42

# Hyperparameters for training/testing
batch_size = 64
num_epochs = 500
samples_per_epoch = 5000
learning_rate = 1.e-3 # Adam
t_his = 25 # number of past motions (c)
t_pred = 100 # number of future motions (t)

# Hyperparaemters for the VAE
lambda_v = 1000
beta = 0.1

# Inference configurations
all_algos = ['vae']
nk = 5 # sample images for reconstruction


def set_seed(seed):
    """Set seed for (1) tensorflow, (2) random and (3) numpy for stable results"""
    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

# TODO: Add the pending terms to the loss function in the conditional VAE for DLow
def loss_function(x, y_rec, y, mu, logvar):
    mse = tf.reduce_mean(tf.pow(y_rec - y, 2))
    kl_divergence = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mu) - tf.exp(logvar))
    loss = mse + beta * kl_divergence
    return loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train')
    args = parser.parse_args()

    # Set the random sets for reproducibility
    set_seed(RANDOM_SEED)

    # Load the train/test splits
    train_ds = DatasetH36M('train', t_his, t_pred, actions='all')
    test_ds = DatasetH36M('test', t_his, t_pred, actions='all')

    #######################################
    # Model training
    #######################################
    if args.mode == 'train':

        # traj_dim -> stacked number of joints for training
        vae = VAE(name='vae',
                  traj_dim = train_ds.traj_dim,
                  t_his = t_his,
                  t_pred = t_pred)
        vae.encoder.summary()
        vae.decoder.summary()

        # Loss function, metrics and optimizer
        #loss = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        training_loss_tracker = tf.keras.metrics.Mean(name="training_loss")

        train_stats = trange(0, desc='training_loss')
        for r in tqdm(range(0, num_epochs)):
            generator = train_ds.sampling_generator(num_samples=samples_per_epoch, batch_size=batch_size)
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
                    mu_enc, logvar_enc = vae.encode(x)
                    z = vae.reparameterize(mu_enc, logvar_enc)
                    y_rec = vae.decode(z)
                    total_loss = loss_function(x, y_rec, y, mu_enc, logvar_enc)

                gradients = tape.gradient(total_loss, vae.trainable_variables)
                optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
                training_loss_tracker.update_state(total_loss)
                train_stats.set_description('training_loss=%g' % training_loss_tracker.result())
                #break
            #break
        # Save the model to disk
        vae.encoder.save(f"models/vae-encoder-{num_epochs}.model")
        vae.decoder.save(f"models/vae-decoder-{num_epochs}.model")

    #######################################
    # Model Inference
    #######################################
    elif args.mode == 'inference':

        vae = VAE(name='vae')
        vae.encoder = models.load_model(f"models/vae-encoder-{num_epochs}.model")
        vae.decoder = models.load_model(f"models/vae-decoder-{num_epochs}.model")
        vae.encoder.summary()
        vae.decoder.summary()

        #######################################
        # List of models to test
        #######################################
        algos = []
        for algo in all_algos:
            algos.append(algo)
        vis_algos = algos.copy()

        def get_prediction(traj_np, algo, sample_num, num_seeds=1, concat_hist=True):

            # Remove the center hip joint for training
            traj_np = traj_np[..., 1:, :]

            # Stack all joints
            traj_np = traj_np.reshape(traj_np.shape[0], traj_np.shape[1], -1)

            # Transpose for selecting frames instead of batches
            traj = np.ascontiguousarray(np.transpose(traj_np, (1, 0, 2)))

            # Transpose back to batches and take past and future motions for encoding
            x = np.transpose(traj[:t_his], (1, 0, 2))
            y = np.transpose(traj[t_his:], (1, 0, 2))

            # Generate multiple samples
            x_multi = tf.repeat(x, repeats = [sample_num * num_seeds], axis=0)

            # Apply the variational autoencoder
            mu_enc, logvar_enc = vae.encode(x_multi)
            z = vae.reparameterize(mu_enc, logvar_enc)
            y_rec_multi = vae.decode(z)
            y_rec_multi = y_rec_multi.numpy()

            # Merge the initial and predicted frames
            merged_frames = np.concatenate((x_multi, y_rec_multi), axis=1)

            if merged_frames.shape[0] > 1:
                merged_frames = merged_frames.reshape(-1, sample_num, merged_frames.shape[-2], merged_frames.shape[-1])
            else:
                merged_frames = merged_frames[None, ...]

            return merged_frames

        def post_process(pred, data):

            # Reshape the stacked poses into individual joints
            pred_joints = pred.reshape(pred.shape[0], pred.shape[1], -1, 3)

            # Extract the center hip joint from the original data (This joint was removed for training)
            center_joint = np.tile(data[..., :1, :], (pred_joints.shape[0], 1, 1, 1))

            # Merge the center joint and predicted ones
            pred = np.concatenate((center_joint, pred_joints), axis=2)

            # Set the center joint to zero to center the image?
            pred[..., :1, :] = 0
            return pred

        def pose_generator():

            while True:
                data = test_ds.sample()

                # gt
                gt = data[0].copy()
                gt[:, :1, :] = 0
                poses = {'context': gt, 'gt': gt}

                # ae
                for algo in vis_algos:
                    pred = get_prediction(data, algo, nk)[0]
                    pred = post_process(pred, data)
                    for i in range(1, pred.shape[0] + 1):
                        poses[f'{algo}_{i}'] = pred[i-1]
                yield poses

        pose_gen = pose_generator()
        render_animation(test_ds.skeleton, pose_gen, vis_algos, t_his, fix_0=True, output=None, size=5, ncol=7, interval=50)

