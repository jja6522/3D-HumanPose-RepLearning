import numpy as np
import os
from utils.dataset import Dataset
from utils.skeleton import Skeleton
from scipy.spatial.transform import Rotation
import copy


class Dataset3DHP(Dataset):

    def __init__(self, mode, t_his=25, t_pred=100, actions='all', use_vel=False):
        self.use_vel = use_vel
        super().__init__(mode, t_his, t_pred, actions)
        if use_vel:
            self.traj_dim += 3

    def prepare_data(self):
        self.data_file = os.path.join('data', 'data_3dhp.npz')
        self.subjects_split = {'train': [1, 5, 6, 7, 8],
                               'test': [2, 3]}
        self.subjects = ['S%d' % x for x in self.subjects_split[self.mode]]
        self.skeleton = Skeleton(parents=[-1,  0,  1,  2,  0,  4,  5,  0,
                                           7,  8,  9,  8, 11, 12,  8, 14, 15],
                                 joints_right=[1, 2, 3, 14, 15, 16],
                                 joints_left=[4, 5, 6, 11, 12, 13])
        self.kept_joints = np.arange(17)
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
                seq = data_s[action].reshape(1, np.shape(data_s[action][:])[0], 28, 3)
                # Take only the valid joints
                joint_idx = [4, 23, 24, 25, 18, 19, 20, 3, 5, 6, 7, 9, 10, 11, 14, 15, 16]
                sample = seq[:, :, joint_idx]

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

                # Save the entries as float for compatiblity with tensorflow
                data_s[action] = sample_norm.astype(np.float32)

            # Center all joints at the hip for compatibility with H36M
            root_positions = {}
            for k in data_s.keys():
                # Keep track of the global position
                root_positions[k] = copy.deepcopy(data_s[k][:,:1])

                # Remove the root from the 3d position
                poses = data_s[k]
                poses = poses - np.tile( poses[:,:1], [1, len(self.kept_joints), 1] )
                data_s[k] = poses

        # Update the preprocessed data
        self.data = data_f


if __name__ == '__main__':
    np.random.seed(0)
    dataset = Dataset3dhp('train')
    generator = dataset.sampling_generator(num_samples=1000, batch_size=8)
    dataset.normalize_data()
    # generator = dataset.iter_generator()
    for data in generator:
        print(data.shape)


