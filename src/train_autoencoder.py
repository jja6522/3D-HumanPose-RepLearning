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

###########################################################
# FIXME: Hack required to enable GPU operations by TF RNN
###########################################################
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

###########################################################
# Global configurations
###########################################################
RANDOM_SEED = 42

# Hyperparameters from the paper DLow
batch_size = 64
num_epochs = 50
samples_per_epoch = 5000
learning_rate = 1.e-3 # Adam

all_algos = ['ae']
traj_dim = 48 # Stacked number of joints for training
t_his = 25 # number of past motions (c)
t_pred = 100 # number of future motions (t)

nk = 5 # sample images for reconstruction


def set_seed(seed):
    """Set seed for (1) tensorflow, (2) random and (3) numpy for stable results"""
    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class AE(keras.Model):
    """Variational autoencoder."""

    def __init__(self, name='Autoencoder'):
        super(AE, self).__init__(name=name)

        # Encoder architecture
        self.encoder = models.Sequential([
            layers.InputLayer(input_shape=(t_his ,traj_dim)),
            layers.GRU(units=128, return_sequences=False),
            layers.RepeatVector(t_pred),
            layers.Dense(300, activation='tanh', name='enc_mlp1'),
            layers.Dense(200, activation='tanh', name='enc_mlp2'),
        ], name="encoder")

        # Decoder architecture
        self.decoder = models.Sequential([
            layers.InputLayer(input_shape=(t_pred, 200, )),
            layers.Dense(300, activation='tanh', name='dec_mlp1'),
            layers.GRU(units=128, return_sequences=True),
            layers.Dense(traj_dim, activation='tanh', name='dec_mlp2'),
        ], name="decoder")

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train')
    args = parser.parse_args()

    # Set the random sets for reproducibility
    set_seed(RANDOM_SEED)

    #Load the dataset
    train_ds = DatasetH36M('train', t_his, t_pred, actions='all')

    # Use the test set for generating samples
    test_ds = DatasetH36M('test', t_his, t_pred, actions='all')

    #######################################
    # Model training
    #######################################
    if args.mode == 'train':

        autoencoder = AE(name='autoencoder')
        autoencoder.encoder.summary()
        autoencoder.decoder.summary()

        # Loss function, metrics and optimizer
        loss = tf.keras.losses.MeanSquaredError()
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
                    x_enc = autoencoder.encode(x)
                    y_rec = autoencoder.decode(x_enc)
                    total_loss = loss(y, y_rec)

                gradients = tape.gradient(total_loss, autoencoder.trainable_variables)
                optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))
                training_loss_tracker.update_state(total_loss)
                train_stats.set_description('training_loss=%g' % training_loss_tracker.result())

        # Save the model to disk
        autoencoder.encoder.save(f"models/encoder-{num_epochs}.model")
        autoencoder.decoder.save(f"models/decoder-{num_epochs}.model")

    #######################################
    # Model Evaluation
    #######################################
    elif args.mode == 'test':

        autoencoder = AE(name='autoencoder')
        autoencoder.encoder = models.load_model(f"models/encoder-{num_epochs}.model")
        autoencoder.decoder = models.load_model(f"models/decoder-{num_epochs}.model")

        # Loss function and metrics
        loss = tf.keras.losses.MeanSquaredError()
        testing_loss_tracker = tf.keras.metrics.Mean(name="testing_loss")

        test_stats = trange(0, desc='testing_loss')
        test_generator = test_ds.sampling_generator(num_samples=samples_per_epoch, batch_size=batch_size)
        for traj_np in test_generator:

            # Remove the center hip joint for training
            traj_np = traj_np[..., 1:, :]

            # Stack all joints
            traj_np = traj_np.reshape(traj_np.shape[0], traj_np.shape[1], -1)

            # Transpose for selecting frames instead of batches
            traj = np.ascontiguousarray(np.transpose(traj_np, (1, 0, 2)))

            # Transpose back to batches and take past and future motions for encoding
            x = np.transpose(traj[:t_his], (1, 0, 2))
            y = np.transpose(traj[t_his:], (1, 0, 2))

            x_enc = autoencoder.encode(x)
            y_rec = autoencoder.decode(x_enc)
            total_loss = loss(y, y_rec)

            testing_loss_tracker.update_state(total_loss)
            test_stats.set_description('testing_loss=%g' % testing_loss_tracker.result())

    #######################################
    # Model Inference
    #######################################
    elif args.mode == 'inference':

        autoencoder = AE(name='autoencoder')
        autoencoder.encoder = models.load_model(f"models/encoder-{num_epochs}.model")
        autoencoder.decoder = models.load_model(f"models/decoder-{num_epochs}.model")

        #######################################
        # Inference
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

            # Apply the autoencoder
            z = autoencoder.encode(x_multi)
            y_rec_multi = autoencoder.decode(z)
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

