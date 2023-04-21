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


RANDOM_SEED = 42

batch_size = 64
num_vae_epoch = 500
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

    # Training mode
    if args.mode == 'train':

        # Encoder architecture
        encoder = models.Sequential([
            layers.Flatten(input_shape=(125, 48)),
            layers.Dense(500, activation='relu', name='input_to_hidden'),
            layers.Dense(50, activation='sigmoid', name='hidden_to_latent')
        ], name='encoder')

        encoder.summary()

        # Decoder architecture
        decoder = models.Sequential([
            layers.Dense(500, activation='relu', input_shape=(50, ), name='latent_to_hidden'),
            layers.Dense(6000, activation='sigmoid', name='hidden_to_output'),
            layers.Reshape((125, 48))
        ], name='decoder')

        decoder.summary()

        # Autoencoder
        autoencoder = models.Model(inputs=encoder.input, outputs=decoder(encoder.output), name='autoencoder')
        autoencoder.summary()

        #######################################
        # Model training
        #######################################

        # Loss function and optmizer
        loss = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam(learning_rate=vae_lr)

        # Metrics
        training_loss_tracker = tf.keras.metrics.Mean(name="training_loss")

        train_stats = trange(0, desc='training_loss')
        for r in tqdm(range(0, num_vae_epoch)):
            generator = train_ds.sampling_generator(num_samples=num_vae_data_sample, batch_size=batch_size)
            for traj_np in generator:
                #print(traj_np.shape)
                traj_np = traj_np[..., 1:, :].reshape(traj_np.shape[0], traj_np.shape[1], -1)
                #print(traj_np.shape)

                with tf.GradientTape() as tape:
                    Y_rec = autoencoder(traj_np, traj_np)
                    total_loss = loss(traj_np, Y_rec)
                gradients = tape.gradient(total_loss, autoencoder.trainable_variables)
                optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))
                training_loss_tracker.update_state(total_loss)
                train_stats.set_description('training_loss=%g' % training_loss_tracker.result())

        # Save the model to disk
        autoencoder.save(f"models/autoencoder-{num_vae_epoch}.model")

    elif args.mode == 'test':

        autoencoder = models.load_model(f"models/autoencoder-{num_vae_epoch}.model")

        # Loss function and metrics
        loss = tf.keras.losses.MeanSquaredError()
        testing_loss_tracker = tf.keras.metrics.Mean(name="testing_loss")

        #######################################
        # Model evaluation
        #######################################
        test_stats = trange(0, desc='testing_loss')
        test_generator = test_ds.sampling_generator(num_samples=num_vae_data_sample, batch_size=batch_size)
        for traj_np in test_generator:
            traj_np = traj_np[..., 1:, :].reshape(traj_np.shape[0], traj_np.shape[1], -1)
            Y_rec = autoencoder.predict(traj_np)
            total_loss = loss(traj_np, Y_rec)
            testing_loss_tracker.update_state(total_loss)
            test_stats.set_description('testing_loss=%g' % testing_loss_tracker.result())

    elif args.mode == 'inference':

        autoencoder = models.load_model(f"models/autoencoder-{num_vae_epoch}.model")

        #######################################
        # Inference
        #######################################
        algos = []
        for algo in all_algos:
            algos.append(algo)
        vis_algos = algos.copy()

        def get_prediction(data, algo, sample_num, num_seeds=1, concat_hist=True):

            # Flatten the sample pose for passing through the network
            traj_np = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
            print(traj_np.shape)

            # Take only the last t_hist frames
            X = traj_np[:t_his]
            print(X.shape)

            # Generate multiple samples
            X = tf.repeat(X, repeats = [sample_num * num_seeds], axis=0)
            print(X.shape)

            # Apply the autoencoder
            Y = autoencoder.predict(X)

            if Y.shape[0] > 1:
                Y = Y.reshape(-1, sample_num, Y.shape[-2], Y.shape[-1])
            else:
                Y = Y[None, ...]
            return Y

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

