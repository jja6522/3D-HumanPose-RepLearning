import numpy as np
import os
from utils.dataset import Dataset
from utils.skeleton import Skeleton
from scipy.spatial.transform import Rotation

# The dataset contains 32 joints but only 16 are movable

# compatible with H36 dataset
joint_idx = [8, 6, 15, 16, 17, 10, 11, 12, 24, 25, 26, 19, 20, 21, 5, 4, 7]

class Dataset3dhp(Dataset):

    def __init__(self, mode, t_his=25, t_pred=100, actions='all', use_vel=False):
        self.use_vel = use_vel
        super().__init__(mode, t_his, t_pred, actions)
        if use_vel:
            self.traj_dim += 3

    def prepare_data(self):
        self.data_file = r"D:\Documents\RIT Course Work\Spring 2023\Neural Computing\Project\3D-HumanPose-RepL-main\src\utils\data_3dhp.npz"
        self.subjects_split = {'train': [1, 5, 6, 7, 8],
                               'test': [2, 3]}
        self.subjects = ['S%d' % x for x in self.subjects_split[self.mode]]
        self.skeleton = Skeleton(parents=[-1, 0, 1, 2, 3,
                              2, 5, 6,
                              2, 8, 9,
                              0, 11, 12,
                              0, 14, 15],
                    joints_left=[5, 6, 7, 11, 12, 13],
                    joints_right=[8, 9, 10, 14, 15, 16])
        self.removed_joints = {17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}
        self.kept_joints = np.array([x for x in range(32) if x not in self.removed_joints])
        self.skeleton.remove_joints(self.removed_joints)
        # self.skeleton._parents[11] = 8
        # self.skeleton._parents[14] = 8
        self.process_data()

    def process_data(self):
        data_o = np.load(self.data_file, allow_pickle=True)['data'].item()
        data_f = dict(filter(lambda x: x[0] in self.subjects, data_o.items()))
        if self.actions != 'all':
            for key in list(data_f.keys()):
                data_f[key] = dict(filter(lambda x: all([a in x[0] for a in self.actions]), data_f[key].items()))
                if len(data_f[key]) == 0:
                    data_f.pop(key)
        for data_s in data_f.values():
            for action in data_s.keys():
                # this is edited to keep the mpi_inf compatible
                seq = data_s[action][600:725].reshape(1, 125, 28, 3)
                # Take only the valid joints
                joints_idx = [4, 3, 5, 6, 7, 9, 10, 11, 14, 15, 16, 18, 19, 20, 23, 24, 25]
                sample = seq[:, :, joints_idx]

                # Rotate 270 degrees all coordinates
                sample_new = sample.reshape(sample.shape[1] * sample.shape[2], 3)

                # Axis and rotation matrix
                axis = [1, 0, 0]
                axis = axis / np.linalg.norm(axis)
                rot = Rotation.from_rotvec(3 * np.pi / 2 * axis)

                # Apply rotation
                sample_rot = rot.apply(sample_new)
                sample_rot = sample_rot.reshape(sample.shape[1], sample.shape[2], 3)

                # standard normalization
                mean = sample_rot.mean()
                std = sample_rot.std()
                sample_norm = (sample_rot - mean) / std

                data_s[action] = sample_norm
        self.data = data_f



if __name__ == '__main__':
    np.random.seed(0)
    dataset = Dataset3dhp('train')
    generator = dataset.sampling_generator(num_samples=1000, batch_size=8)
    dataset.normalize_data()
    # generator = dataset.iter_generator()
    for data in generator:
        print(data.shape)


