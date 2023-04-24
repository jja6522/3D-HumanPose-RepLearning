import os
import numpy as np
from utils.utils_3dhp import *

import scipy.io as scio

# change path to root directory of dataset
data_path=r'D:\Documents\RIT Course Work\Spring 2023\Neural Computing\Project\dataset used'

cam_set = [0, 1, 2, 4, 5, 6, 7, 8]
joint_set = [8, 6, 15, 16, 17, 10, 11, 12, 24, 25, 26, 19, 20, 21, 5, 4, 7]
# joint_set = [7, 5, 14, 15, 16, 9, 10, 11, 23, 24, 25, 18, 19, 20, 4, 3, 6]

dic_seq={}


for root, dirs, files in os.walk(data_path):

    for file in files:
        if file.endswith("mat"):

            path = root.split("\\")
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









