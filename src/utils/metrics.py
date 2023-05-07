import tensorflow as tf
import numpy as np

from scipy.spatial.distance import pdist
from scipy.spatial.distance import pdist, squareform

#FIXME: Investigate why the animation is a bit slow at the beginning when using this function.
#       Attempt converting it to @tf.function since the decoding from the cvae may be the slow part
def dlow_sampling(traj_np, model, cvae, t_his, sample_num, num_seeds=1, concat_hist=True):
    # Remove the center hip joint for training
    traj_np = traj_np[..., 1:, :]

    # Stack all joints
    traj_np = traj_np.reshape(traj_np.shape[0], traj_np.shape[1], -1)

    # Transpose for selecting frames instead of batches
    traj = np.ascontiguousarray(np.transpose(traj_np, (1, 0, 2)))

    # Transpose back to batches and take conditioned motions c
    c = np.transpose(traj[:t_his], (1, 0, 2))

    # Use dlow encoder to get the affine transformation params
    z = model.sample_prior(c)
    z = tf.reshape(z, [z.shape[0] * z.shape[1], z.shape[2]])

    # Take the latent codes to be decoded
    z_sample = z[:sample_num, :]

    # Repeat the conditional input c for k samples
    c_mul = tf.repeat(c, repeats = [sample_num], axis=1)
    c_mul = tf.reshape(c_mul, [c.shape[0] * sample_num, c.shape[1], c.shape[2]])

    # Decode the generated samples using the cvae decoder
    x_mul_new = cvae.decode(z_sample, c_mul)
    x_mul_new = x_mul_new.numpy()

    # Merge the past motions c with the future predicted motions y
    if concat_hist:
        x_mul_new = np.concatenate((c_mul, x_mul_new), axis=1)

    if x_mul_new.shape[0] > 1:
        x_mul_new = x_mul_new.reshape(-1, sample_num, x_mul_new.shape[-2], x_mul_new.shape[-1])
    else:
        x_mul_new = x_mul_new[None, ...]

    return x_mul_new


"""Adapted sampling taken from https://github.com/Khrylx/DLow/blob/master/motion_pred/eval.py"""
def random_sampling(traj_np, model, t_his, sample_num, num_seeds=1, concat_hist=True):
    # Remove the center hip joint for training
    traj_np = traj_np[..., 1:, :]

    # Stack all joints
    traj_np = traj_np.reshape(traj_np.shape[0], traj_np.shape[1], -1)

    # Transpose for selecting frames instead of batches
    traj = np.ascontiguousarray(np.transpose(traj_np, (1, 0, 2)))

    # Transpose back to batches and take conditioned motions c
    c = np.transpose(traj[:t_his], (1, 0, 2))

    # Repeat the pose for the number of samples
    c_mul = tf.repeat(c, repeats = [sample_num * num_seeds], axis=0)

    # Take a sample from the latent space
    x_mul_new = model.sample_prior(c_mul)
    x_mul_new = x_mul_new.numpy()

    # Merge the past motions c with the future predicted motions y
    if concat_hist:
        x_mul_new = np.concatenate((c_mul, x_mul_new), axis=1)

    if x_mul_new.shape[0] > 1:
        x_mul_new = x_mul_new.reshape(-1, sample_num, x_mul_new.shape[-2], x_mul_new.shape[-1])
    else:
        x_mul_new = x_mul_new[None, ...]

    return x_mul_new


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

