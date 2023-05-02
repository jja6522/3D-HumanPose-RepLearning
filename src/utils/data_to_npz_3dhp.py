import os
import numpy as np
from utils.utils_3dhp import *

import scipy.io as scio

# change path to root directory of dataset
data_path='/home/johann/datasets/mpi_inf_3dhp/'

cam_set = [0, 1, 2, 4, 5, 6, 7, 8]

all_joint_names = ['spine3', 'spine4', 'spine2', 'spine', 'pelvis',
                   'neck', 'head', 'head_top', 'left_clavicle', 'left_shoulder', 'left_elbow',
                   'left_wrist', 'left_hand',  'right_clavicle', 'right_shoulder', 'right_elbow', 'right_wrist',
                   'right_hand', 'left_hip', 'left_knee', 'left_ankle', 'left_foot', 'left_toe',
                   'right_hip' , 'right_knee', 'right_ankle', 'right_foot', 'right_toe']

MPI_3DHP_JOINTS = dict(zip(all_joint_names, range(0, 28)))

# Joint indexes for compatibility with h36m. The original hm36 has 32 joints and only 17 are taken.
joint_set = [4, 3, 5, 6, 7, 9, 10, 11, 14, 15, 16, 18, 19, 20, 23, 24, 25]

dic_seq={}


for root, dirs, files in os.walk(data_path):

    for file in files:
        if file.endswith("mat"):

            path = root.split("/")
            subject = path[-2][1]

            seq = path[-1][3]
            print("loading %s %s..."%(path[-2],path[-1]))

            temp = mpii_get_sequence_info(subject, seq)

            frames = temp[0]
            fps = temp[1]

            data = scio.loadmat(os.path.join(root, file))

            data_3d = data['univ_annot3'][[0]]

            data_3d_cam = data_3d[0][0]

            data_3d_cam = data_3d_cam.reshape(data_3d_cam.shape[0], 28,3)

            data_3d_select = data_3d_cam[:frames, joint_set]

            if path[-2] not in dic_seq:
                dic_seq[path[-2]] = {}
            dic_seq[path[-2]][path[-1]] = data_3d_select

np.savez_compressed('data_3dhp', data=dic_seq)

