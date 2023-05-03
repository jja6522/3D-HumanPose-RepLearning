import tensorflow as tf
import numpy as np

from scipy.spatial.distance import pdist
from scipy.spatial.distance import pdist, squareform

"""Adapted sampling taken from https://github.com/Khrylx/DLow/blob/master/motion_pred/eval.py"""

def sampling(traj_np, model, t_his, sample_num, num_seeds=1, concat_hist=True):
    # Remove the center hip joint for training
    traj_np = traj_np[..., 1:, :]

    # Stack all joints
    traj_np = traj_np.reshape(traj_np.shape[0], traj_np.shape[1], -1)

    # Transpose for selecting frames instead of batches
    traj = np.ascontiguousarray(np.transpose(traj_np, (1, 0, 2)))

    # Transpose back to batches and take past and future motions for encoding
    x = np.transpose(traj[:t_his], (1, 0, 2))

    # Repeat the pose for the number of samples
    x_mul = tf.repeat(x, repeats = [sample_num * num_seeds], axis=0)

    # Take a random sample from the latent space
    y_mul_new = model.sample(x_mul)
    y_mul_new = y_mul_new.numpy()

    # Merge the past motions c with the future predicted motions y
    if concat_hist:
        x_mul = x_mul.numpy()
        y_mul_new = np.concatenate((x_mul, y_mul_new), axis=1)

    if y_mul_new.shape[0] > 1:
        y_mul_new = y_mul_new.reshape(-1, sample_num, y_mul_new.shape[-2], y_mul_new.shape[-1])
    else:
        y_mul_new = y_mul_new[None, ...]

    return y_mul_new


""" Metrics taken from https://github.com/Khrylx/DLow/blob/master/motion_pred/eval.py"""

def compute_diversity(pred, *args):
    if pred.shape[0] == 1:
        return 0.0
    dist = pdist(pred.reshape(pred.shape[0], -1))
    diversity = dist.mean().item()
    return diversity


def compute_ade(pred, gt, *args):
    diff = pred - gt
    dist = np.linalg.norm(diff, axis=2).mean(axis=1)
    return dist.min()


def compute_fde(pred, gt, *args):
    diff = pred - gt
    dist = np.linalg.norm(diff, axis=2)[:, -1]
    return dist.min()


def compute_mmade(pred, gt, gt_multi):
    gt_dist = []
    for gt_multi_i in gt_multi:
        dist = compute_ade(pred, gt_multi_i)
        gt_dist.append(dist)
    gt_dist = np.array(gt_dist).mean()
    return gt_dist


def compute_mmfde(pred, gt, gt_multi):
    gt_dist = []
    for gt_multi_i in gt_multi:
        dist = compute_fde(pred, gt_multi_i)
        gt_dist.append(dist)
    gt_dist = np.array(gt_dist).mean()
    return gt_dist


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

