import os
import numpy as np
from utils.utils_3dhp import *
import time
import argparse
import scipy.io as scio


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to mpi_inf_3dhp dataset")
    parser.add_argument("--output_path", default="data", help="Path to store the preprocessed data")
    args = parser.parse_args()

    dic_seq={}

    for root, dirs, files in os.walk(args.data_path):

        for file in files:

            if file.endswith("mat"):

                path = root.split(os.sep)
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

                data_3d_select = data_3d_cam[:frames, :]

                if path[-2] not in dic_seq:
                    dic_seq[path[-2]] = {}
                dic_seq[path[-2]][path[-1]] = data_3d_select

    # Save the pre-procesed dataset
    np.savez_compressed(args.output_path + '/data_3dhp', data=dic_seq)

