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

batch_size = 64
num_vae_epoch = 10
num_vae_data_sample = 5000
vae_lr = 1.e-3
all_algos = ['ae']
nk = 5
t_his = 25
t_pred = 100


def set_seed(seed):
    """Set seed for (1) tensorflow, (2) random and (3) numpy for stable results"""
    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class AE(keras.Model):
    """Variational autoencoder."""

    def __init__(self, name='Autoencoder'):
        super(AE, self).__init__(name=name)

        self.encoder = models.Sequential([
            layers.InputLayer(input_shape=(125, 48, )),
            layers.GRU(units=128, return_sequences=True)
        ], name="encoder")

        self.decoder = models.Sequential([
            layers.InputLayer(input_shape=(125, 128, )),
            layers.GRU(units=128, return_sequences=True),
            layers.Dense(48)
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
        optimizer = tf.keras.optimizers.Adam(learning_rate=vae_lr)
        training_loss_tracker = tf.keras.metrics.Mean(name="training_loss")

        train_stats = trange(0, desc='training_loss')
        for r in tqdm(range(0, num_vae_epoch)):
            generator = train_ds.sampling_generator(num_samples=num_vae_data_sample, batch_size=batch_size)
            for x in generator:
                #print(traj_np.shape)
                x = x[..., 1:, :].reshape(x.shape[0], x.shape[1], -1)
                #print(traj_np.shape)

                with tf.GradientTape() as tape:
                    z = autoencoder.encode(x)
                    y_rec = autoencoder.decode(z)
                    total_loss = loss(x, y_rec)

                gradients = tape.gradient(total_loss, autoencoder.trainable_variables)
                optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))
                training_loss_tracker.update_state(total_loss)
                train_stats.set_description('training_loss=%g' % training_loss_tracker.result())

        # Save the model to disk
        autoencoder.encoder.save(f"models/encoder-{num_vae_epoch}.model")
        autoencoder.decoder.save(f"models/decoder-{num_vae_epoch}.model")

    #######################################
    # Model Evaluation
    #######################################
    elif args.mode == 'test':

        autoencoder = AE(name='autoencoder')
        autoencoder.encoder = models.load_model(f"models/encoder-{num_vae_epoch}.model")
        autoencoder.decoder = models.load_model(f"models/decoder-{num_vae_epoch}.model")

        # Loss function and metrics
        loss = tf.keras.losses.MeanSquaredError()
        testing_loss_tracker = tf.keras.metrics.Mean(name="testing_loss")

        test_stats = trange(0, desc='testing_loss')
        test_generator = test_ds.sampling_generator(num_samples=num_vae_data_sample, batch_size=batch_size)
        for x in test_generator:
            x = x[..., 1:, :].reshape(x.shape[0], x.shape[1], -1)

            # Evaluat without updating gradients
            z = autoencoder.encode(x)
            y_rec = autoencoder.decode(z)
            total_loss = loss(x, y_rec)

            testing_loss_tracker.update_state(total_loss)
            test_stats.set_description('testing_loss=%g' % testing_loss_tracker.result())

    #######################################
    # Model Inference
    #######################################
    elif args.mode == 'inference':

        autoencoder = AE(name='autoencoder')
        autoencoder.encoder = models.load_model(f"models/encoder-{num_vae_epoch}.model")
        autoencoder.decoder = models.load_model(f"models/decoder-{num_vae_epoch}.model")

        #######################################
        # Inference
        #######################################
        algos = []
        for algo in all_algos:
            algos.append(algo)
        vis_algos = algos.copy()

        def get_prediction(data, algo, sample_num, num_seeds=1, concat_hist=True):

            # Flatten the sample pose for passing through the network
            x = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
            print(x.shape)

            # Take only the last t_hist frames
            x = x[:t_his]
            print(x.shape)

            # Generate multiple samples
            x_multi = tf.repeat(x, repeats = [sample_num * num_seeds], axis=0)
            print(x_multi.shape)

            # Apply the autoencoder
            z = autoencoder.encode(x_multi)
            y_rec_multi = autoencoder.decode(z)
            y_rec_multi = y_rec_multi.numpy()
            print(y_rec_multi.shape)

            if y_rec_multi.shape[0] > 1:
                y_rec_multi = y_rec_multi.reshape(-1, sample_num, y_rec_multi.shape[-2], y_rec_multi.shape[-1])
            else:
                y_rec_multi = y_rec_multi[None, ...]
            return y_rec_multi

        def post_process(pred, data):
            # Unflatten reconstructed poses for evaluation
            pred = pred.reshape(pred.shape[0], pred.shape[1], -1, 3)
            pred = np.concatenate((np.tile(data[..., :1, :], (pred.shape[0], 1, 1, 1)), pred), axis=2)
            pred[..., :1, :] = 0
            return pred

        def pose_generator():

            while True:
                # Take a test sample and test its reconstructed pose
                sample_pose = test_ds.sample()
                print("Sampled Pose:", sample_pose.shape)

                # gt
                gt = sample_pose[0].copy()
                gt[:, :1, :] = 0
                poses = {'context': gt, 'gt': gt}

                # ae
                for algo in vis_algos:
                    pred = get_prediction(sample_pose, algo, nk)[0]
                    print(pred.shape)
                    pred = post_process(pred, sample_pose)
                    print(pred.shape)
                    for i in range(1, pred.shape[0] + 1):
                        poses[f'{algo}_{i}'] = pred[i-1]
                yield poses

        pose_gen = pose_generator()

        render_animation(test_ds.skeleton, pose_gen, vis_algos, t_his, fix_0=True, output=None, size=5, ncol=7, interval=50)

