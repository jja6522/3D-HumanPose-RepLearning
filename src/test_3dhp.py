import sys
import random
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers

import time
import argparse
from tqdm import tqdm, trange
import csv
import os

from scipy.spatial.distance import pdist
from scipy.spatial.distance import pdist, squareform

from utils.dataset_h36m import DatasetH36M
from utils.dataset_3dhp import Dataset3dhp
from utils.metrics import random_sampling, dlow_sampling, compute_diversity, compute_ade, compute_fde, compute_mmade, compute_mmfde, AverageMeter
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

# General Hyperparameters for training/testing
t_his = 25 # number of past motions (c)
t_pred = 100 # number of future motions (x)


def set_seed(seed):
    """Set seed for (1) tensorflow, (2) random and (3) numpy for stable results"""
    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_gt(data):
    gt = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
    return gt[:, t_his:, :]


def get_multimodal_gt(dataset, t_his, threshold):
    all_data = []
    data_gen = dataset.iter_generator(step=t_his)
    for data in data_gen:
        data = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
        all_data.append(data)
    all_data = np.concatenate(all_data, axis=0)
    all_start_pose = all_data[:, t_his - 1, :]
    pd = squareform(pdist(all_start_pose))
    traj_gt_arr = []
    for i in range(pd.shape[0]):
        ind = np.nonzero(pd[i] < threshold)
        traj_gt_arr.append(all_data[ind][:, t_his:, :])
    return traj_gt_arr


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ae", help="all, ae, vae, dlow")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs to load a model")
    parser.add_argument("--dlow_samples", type=int, default=10, help="Number of DLow samples for epsilon (nk)")
    parser.add_argument('--num_seeds', type=int, default=1)
    parser.add_argument('--multimodal_threshold', type=float, default=0.5)
    args = parser.parse_args()

    # Set the random sets for reproducibility
    set_seed(RANDOM_SEED)

    #######################################
    # Dataset loading
    #######################################
    test_ds = Dataset3dhp('test', t_his, t_pred, actions='all')
    traj_gt_arr = get_multimodal_gt(test_ds, t_his, args.multimodal_threshold)

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

    #######################################
    # Model testing
    #######################################
    stats_func = {'Diversity': compute_diversity, 'ADE': compute_ade,
                  'FDE': compute_fde, 'MMADE': compute_mmade, 'MMFDE': compute_mmfde}
    stats_names = list(stats_func.keys())
    stats_meter = {x: {y: AverageMeter() for y in model_dict.keys()} for x in stats_names}

    data_gen = test_ds.iter_generator(step=t_his)
    num_samples = 0
    for i, data in enumerate(data_gen):
        num_samples += 1
        gt = get_gt(data)
        gt_multi = traj_gt_arr[i]

        for model in eval_models:

            if model.name == 'ae' or model.name == 'vae':
                pred = random_sampling(data, model, t_his, args.dlow_samples, args.num_seeds, concat_hist=False)

            # Sampling dlow requires the cvae decoder
            elif model.name == 'dlow':
                pred = dlow_sampling(data, model, model_dict['vae'], t_his, args.dlow_samples, args.num_seeds, concat_hist=False)

            for stats in stats_names:
                val = 0
                for pred_i in pred:
                    val += stats_func[stats](pred_i, gt, gt_multi) / args.num_seeds
                stats_meter[stats][model.name].update(val)

        print('-' * 80)
        for stats in stats_names:
            str_stats = f'{num_samples:04d} {stats}: ' + ' '.join([f'{x}: {y.val:.4f}({y.avg:.4f})' for x, y in stats_meter[stats].items()])
            print(str_stats)

    # Save results for all metrics to a file
    results_file = 'results/stats_%s_nk%s_epochs%s.csv' % (args.model, str(args.dlow_samples), str(args.num_epochs))
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['Metric'] + list(model_dict.keys()))
        writer.writeheader()
        for stats, meter in stats_meter.items():
            new_meter = {x: y.avg for x, y in meter.items()}
            new_meter['Metric'] = stats
            writer.writerow(new_meter)

