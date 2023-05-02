import numpy as np
import os
from utils.dataset import Dataset
from utils.skeleton import Skeleton


class Dataset3dhp(Dataset):

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
        self.skeleton = Skeleton(parents=[-1, 0, 1, 2, 3,
                                          2, 5, 6,
                                          2, 8, 9,
                                          0, 11, 12,
                                          0, 14, 15],
                                joints_left=[5, 6, 7, 11, 12, 13],
                                joints_right=[8, 9, 10, 14, 15, 16])
        self.kept_joints = np.arange(17)
        self.process_data()

    #FIXME: This part needs to be adapted for the new dataset!
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
                seq = data_s[action][:, self.kept_joints, :]
                if self.use_vel:
                    v = (np.diff(seq[:, :1], axis=0) * 50).clip(-5.0, 5.0)
                    v = np.append(v, v[[-1]], axis=0)
                seq[:, 1:] -= seq[:, :1]
                if self.use_vel:
                    seq = np.concatenate((seq, v), axis=1)
                data_s[action] = seq
        self.data = data_f


if __name__ == '__main__':
    np.random.seed(0)
    dataset = Dataset3dhp('train')
    generator = dataset.sampling_generator(num_samples=1000, batch_size=8)
    dataset.normalize_data()
    # generator = dataset.iter_generator()
    for data in generator:
        print(data.shape)


