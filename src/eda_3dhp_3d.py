import pandas as pd
import numpy as np
import scipy.io as sio
import os
from utils.dataset_h36m import DatasetH36M
from utils.dataset_3dhp import Dataset3dhp
from utils.visualization import render_animation
from utils.skeleton import Skeleton
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
from scipy.spatial.transform import Rotation


# Load the data from a matlab file
data = sio.loadmat('/home/johann/datasets/mpi_inf_3dhp/S3/Seq1/annot.mat')

# mpi-inf-3dhp format
# The dataset contains 28 joints
all_joint_names = ['spine3', 'spine4', 'spine2', 'spine', 'pelvis', 
                   'neck', 'head', 'head_top', 'left_clavicle', 'left_shoulder', 'left_elbow',
                   'left_wrist', 'left_hand',  'right_clavicle', 'right_shoulder', 'right_elbow', 'right_wrist',
                   'right_hand', 'left_hip', 'left_knee', 'left_ankle', 'left_foot', 'left_toe',
                   'right_hip' , 'right_knee', 'right_ankle', 'right_foot', 'right_toe']

MPI_3DHP_JOINTS = dict(zip(all_joint_names, range(0, 28)))

# Take a sample
m3dhp = data['univ_annot3'][0][0][1000].reshape(28, 3)

# Render the complete skeleton positions in a 3d plot
fig = plt.figure()
ax = plt.axes(projection='3d')

spinex = m3dhp[:,0][[4, 5, 6, 7]]
spiney = m3dhp[:,1][[4, 5, 6, 7]]
spinez = m3dhp[:,2][[4, 5, 6, 7]]
ax.plot(spinex, spiney, spinez, marker='*', color='orange')

spine2x = m3dhp[:,0][[0, 1, 2, 3]]
spine2y = m3dhp[:,1][[0, 1, 2, 3]]
spine2z = m3dhp[:,2][[0, 1, 2, 3]]
ax.plot(spine2x, spine2y, spine2z, marker='o')

left_legx = m3dhp[:,0][[4, 18, 19, 20, 21, 22]]
left_legy = m3dhp[:,1][[4, 18, 19, 20, 21, 22]]
left_legz = m3dhp[:,2][[4, 18, 19, 20, 21, 22]]
ax.plot(left_legx, left_legy, left_legz, marker=matplotlib.markers.CARETLEFTBASE, color='blue')

right_legx = m3dhp[:,0][[4, 23, 24, 25, 26, 27]]
right_legy = m3dhp[:,1][[4, 23, 24, 25, 26, 27]]
right_legz = m3dhp[:,2][[4, 23, 24, 25, 26, 27]]
ax.plot(right_legx, right_legy, right_legz, marker=matplotlib.markers.CARETRIGHTBASE, color='blue')

left_armx = m3dhp[:,0][[5, 8, 9, 10, 11, 12]]
left_army = m3dhp[:,1][[5, 8, 9, 10, 11, 12]]
left_armz = m3dhp[:,2][[5, 8, 9, 10, 11, 12]]
ax.plot(left_armx, left_army, left_armz, marker=matplotlib.markers.CARETLEFTBASE, color='green')

right_armx = m3dhp[:,0][[5, 13, 14, 15, 16, 17]]
right_army = m3dhp[:,1][[5, 13, 14, 15, 16, 17]]
right_armz = m3dhp[:,2][[5, 13, 14, 15, 16, 17]]
ax.plot(right_armx, right_army, right_armz, marker=matplotlib.markers.CARETRIGHTBASE, color='green')

plt.show()


## Human3.6m compatible joint set in Our order.
## The O1 and O2 indices are relaive to the joint_idx, regardless of the joint set 
#joint_idx = [7, 5, 14, 15, 16, 9, 10, 11, 23, 24, 25, 18, 19, 20, 4, 3, 6]
#joint_parents_o1 = [ 2, 16, 2, 3, 4, 2, 6, 7, 15, 9, 10, 15, 12, 13, 15, 15, 2]
#joint_parents_o2 = [ 16, 15, 16, 2, 3, 16, 2, 6, 16, 15, 9, 16, 15, 12, 15, 15, 16]

# Render the a subset of joints compatible with h36m
fig = plt.figure()
ax = plt.axes(projection='3d')

spinex = m3dhp[:,0][[4, 3, 5, 6, 7]]
spiney = m3dhp[:,1][[4, 3, 5, 6, 7]]
spinez = m3dhp[:,2][[4, 3, 5, 6, 7]]
ax.plot(spinex, spiney, spinez, marker='*', color='orange')

left_legx = m3dhp[:,0][[4, 18, 19, 20]]
left_legy = m3dhp[:,1][[4, 18, 19, 20]]
left_legz = m3dhp[:,2][[4, 18, 19, 20]]
ax.plot(left_legx, left_legy, left_legz, marker=matplotlib.markers.CARETLEFTBASE, color='blue')

right_legx = m3dhp[:,0][[4, 23, 24, 25]]
right_legy = m3dhp[:,1][[4, 23, 24, 25]]
right_legz = m3dhp[:,2][[4, 23, 24, 25]]
ax.plot(right_legx, right_legy, right_legz, marker=matplotlib.markers.CARETRIGHTBASE, color='blue')

left_armx = m3dhp[:,0][[5, 9, 10, 11]]
left_army = m3dhp[:,1][[5, 9, 10, 11]]
left_armz = m3dhp[:,2][[5, 9, 10, 11]]
ax.plot(left_armx, left_army, left_armz, marker=matplotlib.markers.CARETLEFTBASE, color='green')

right_armx = m3dhp[:,0][[5, 14, 15, 16]]
right_army = m3dhp[:,1][[5, 14, 15, 16]]
right_armz = m3dhp[:,2][[5, 14, 15, 16]]
ax.plot(right_armx, right_army, right_armz, marker=matplotlib.markers.CARETRIGHTBASE, color='green')

plt.show()

# Original indexed at 1
#joint_parents_o1 = [3, 1, 4, 5, 5, 2, 6, 7, 6, 9, 10, 11, 12, 6, 14, 15, 16, 17, 5, 19, 20, 21, 22, 5, 24, 25, 26, 27 ]

# Converted indexed at 0
#joint_parents_o1 = [2, 0, 3, 4, 4, 1, 5, 6, 5, 8, 9, 10, 11, 5, 13, 14, 15, 16, 4, 18, 19, 20, 21, 4, 23, 24, 25, 26]

# Initial skeleton with all the joints
#skeleton = Skeleton(parents=[ 3, 2, 1, 4, -1, 3, 5, 6, # 0-7
#                              5, 5, 9, 10, 11, # 8-12
#                              5, 5, 14, 15, 16, # 13-17
#                              4, 18, 19, 20, 21, # 18-22
#                              4, 23, 24, 25, 26], # 23-27
#                    joints_left=[8, 9, 10, 11, 12, 18, 19, 20, 21, 22],
#                    joints_right=[13, 14, 15, 16, 17, 23, 24, 25, 26, 27])

# Updated skeleton only with joints compatible with h36m
skeleton = Skeleton(parents=[-1, 0, 1, 2, 3,
                              2, 5, 6,
                              2, 8, 9,
                              0, 11, 12,
                              0, 14, 15],
                    joints_left=[5, 6, 7, 11, 12, 13],
                    joints_right=[8, 9, 10, 14, 15, 16])

def pose_generator():
    while True:
        # Get the 28 joints in pairs of 3
        sample = data['univ_annot3'][0][0][600:725].reshape(1, 125, 28, 3)

        # Take only the valid joints
        joints_idx = [4, 3, 5, 6, 7, 9, 10, 11, 14, 15, 16, 18, 19, 20, 23, 24, 25]
        sample = sample[:,:, joints_idx]

        # Rotate 270 degrees all coordinates
        sample_new = sample.reshape(sample.shape[1] * sample.shape[2], 3)

        # Axis and rotation matrix
        axis = [1, 0, 0]
        axis = axis / np.linalg.norm(axis)
        rot = Rotation.from_rotvec(3 * np.pi/2 * axis)

        # Apply rotation
        sample_rot = rot.apply(sample_new)
        sample_rot = sample_rot.reshape(sample.shape[1], sample.shape[2], 3)

        # standard normalization
        mean = sample_rot.mean()
        std = sample_rot.std()
        sample_norm = (sample_rot - mean) / std

        # gt
        gt = sample_norm.copy()
        poses = {'context': gt, 'gt': gt}
        yield poses

pose_gen = pose_generator()

render_animation(skeleton, pose_gen, 'test', 25, fix_0=True, output=None, size=8, ncol=2, interval=50)

