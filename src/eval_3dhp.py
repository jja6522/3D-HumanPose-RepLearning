import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers

import sys
import random
import numpy as np
import matplotlib.pyplot as plt

from utils.dataset_h36m import DatasetH36M
from utils.dataset_3dhp import Dataset3dhp
from utils.visualization import render_animation
from utils.metrics import random_sampling, dlow_sampling

import time
import argparse
from tqdm import tqdm, trange

from models import AE, VAE, DLow

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

# Hyperparameters for model evaluation
t_his = 25 # number of past motions (c)
t_pred = 100 # number of future motions (x)


def set_seed(seed):
    """Set seed for (1) tensorflow, (2) random and (3) numpy for stable results"""
    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def reconstruct(traj_np, model, sample_num, num_seeds=1, concat_hist=True):

    # Remove the center hip joint for training
    traj_np = traj_np[..., 1:, :]

    # Stack all joints
    traj_np = traj_np.reshape(traj_np.shape[0], traj_np.shape[1], -1)

    # Transpose for selecting frames instead of batches
    traj = np.ascontiguousarray(np.transpose(traj_np, (1, 0, 2)))

    # Transpose back to batches and take past and future motions for encoding
    c = np.transpose(traj[:t_his], (1, 0, 2))
    x = np.transpose(traj[t_his:], (1, 0, 2))

    # Repeat the pose for the number of samples
    c_mul = tf.repeat(c, repeats = [sample_num * num_seeds], axis=0)
    x_mul = tf.repeat(x, repeats = [sample_num * num_seeds], axis=0)

    if model.name == 'ae':
        z = model.encode(x_mul, c_mul)
        x_rec_mul = model.decode(z, c_mul)
    elif model.name == 'vae':
        mu, logvar = model.encode(x_mul, c_mul)
        z = model.reparameterize(mu, logvar)
        x_rec_mul = model.decode(z, c_mul)

    # Merge the past motions c with the future predicted motions x
    c_mul = c_mul.numpy()
    x_rec_mul = x_rec_mul.numpy()
    merged_frames = np.concatenate((c_mul, x_rec_mul), axis=1)

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

    return pred


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="dlow", help="all, ae, vae, dlow")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs to load a model")
    parser.add_argument("--dlow_samples", type=int, default=10, help="Number of DLow samples for epsilon (nk)")
    parser.add_argument("--vis_samples", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--action", default= "sampling", help="reconstruct, sampling")
    args = parser.parse_args()

    # Set the random sets for reproducibility
    set_seed(RANDOM_SEED)

    #######################################
    # Dataset loading
    #######################################
    test_ds = Dataset3dhp('test', t_his, t_pred, actions='all')

    # Define the available models
    model_dict = {
                  "ae": AE(name='ae'),
                  "vae": VAE(name='vae'),
                  "dlow": DLow(name='dlow')
                  }

    #######################################
    # Model to be considered
    #######################################
    if args.model == 'all':
        eval_models = model_dict.values()
        print(eval_models)
    else:
        eval_models = [model_dict[args.model]]
        print(eval_models)

    for model in eval_models:
        # Training Dlow requires a pre-trained VAE and dlow_samples
        if model.name == 'dlow':
            model.load_model(args.num_epochs, args.dlow_samples)
            print(">>>>> Dlow was trained with dlow_samples =", model.dlow_samples)
            print(">>>>> Loading pre-trained VAE for DLow", type(model_dict["vae"]))
            model_dict["vae"].load_model(args.num_epochs)
            model_dict["vae"].summary()
        # AE and VAE need only epochs for a pre-trained model
        else:
            model.load_model(args.num_epochs)
        model.summary()


    def pose_generator():
        while True:
            data = test_ds.sample()
            # gt
            gt = data[0].copy()
            poses = {'context': gt, 'gt': gt}
            # models
            for model in eval_models:

                if args.action == 'reconstruct':

                    if model.name == 'ae' or model.name == 'vae':
                        pred = reconstruct(data, model, args.vis_samples)[0]
                    else:
                        print("Reconstruction is not supported for DLow sampling model")
                        continue

                elif args.action == 'sampling':

                    if model.name == 'ae' or model.name == 'vae':
                        pred = random_sampling(data, model, t_his, args.vis_samples)[0]

                    # Sampling dlow requires the cvae decoder
                    elif model.name == 'dlow':
                        pred = dlow_sampling(data, model, model_dict["vae"], t_his, args.vis_samples)[0]

                pred = post_process(pred, data)
                for i in range(1, pred.shape[0] + 1):
                    poses[f'{model.name}_{i}'] = pred[i-1]
            yield poses

    # Invoke the post generator and rendering
    pose_gen = pose_generator()
    model_names = [model.name for model in eval_models]
    render_animation(test_ds.skeleton, pose_gen, model_names, t_his, fix_0=True, output=None, size=5, ncol=7, interval=50)

