import pandas as pd
import numpy as np
import h5py
import imageio.v3 as iio
import matplotlib.pyplot as plt
from utils.dataset_h36m import DatasetH36M
from utils.visualization import render_animation
import time

#np.random.seed(0)

# Subjects : ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
# Actions : ['Eating 2', 'Greeting', 'Sitting 1', 'Waiting', 'Posing', 'Directions', 'SittingDown 2', 
#            'WalkDog', 'Eating', 'Posing 1', 'WalkTogether 1', 'Walking', 'Phoning 1', 'Smoking', 
#            'Photo 1', 'Waiting 1', 'Greeting 1', 'Directions 1', 'Sitting 2', 'WalkTogether', 
#            'Discussion', 'Discussion 1', 'Purchases', 'Phoning', 'Photo', 'WalkDog 1', 'Walking 1', 
#            'SittingDown', 'Smoking 1', 'Purchases 1']

dataset = DatasetH36M('train', actions={'Smoking'})

def pose_generator():
    while True:
        data = dataset.sample()
        # gt
        gt = data[0].copy()
        gt[:, :1, :] = 0
        poses = {'context': gt, 'gt': gt}
        yield poses

pose_gen = pose_generator()

render_animation(dataset.skeleton, pose_gen, 'test', 25, fix_0=True, output=None, size=8, ncol=2, interval=50)
