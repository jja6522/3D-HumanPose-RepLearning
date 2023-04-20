import pandas as pd
import numpy as np
import h5py
import imageio.v3 as iio
import matplotlib.pyplot as plt

# root dir for h36m dataset
H36M_ROOT = '/home/johann/datasets/h36m/'

# Joint names order
H36M_JOINT_NAMES = ['Hip', 'LHip', 'LKnee', 'LFoot', 'RHip', 'RKnee', 'RFoot',
                    'Spine', 'Thorax', 'Neck/Nose', 'Head',
                    'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']

def plot_2d_pose(imagefile, center, joints):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    im = iio.imread(H36M_ROOT + 'images/' + imagefile)
    ax.imshow(im)

    # Joints
    for (k, joint) in enumerate(joints):
        x, y = joint[0], joint[1]
        ax.plot(x, y, 'o', color='b', markersize=5)
        ax.annotate(H36M_JOINT_NAMES[k], xy=(x, y), xytext=(x, y + 25))
    plt.show()

# Read the image annotations from h5
h5_label_file = h5py.File(H36M_ROOT + 'annot/train.h5', 'r')
data_dict = {key: list(value) for key, value in h5_label_file.items()}
df = pd.DataFrame.from_dict(data_dict)

# Read the image files from txt
df['imagefile'] = pd.read_csv(H36M_ROOT + 'annot/train_images.txt', header=None)

df['subj'] = df['imagefile'].apply(lambda x: x.split('.')[0].split('_')[0][-1])
df['cam'] = df['imagefile'].apply(lambda x: x.split('.')[1].split('_')[0])
df['action'] = df['imagefile'].apply(lambda x: '_'.join(x.split('.')[0].split('_')[1:]))

#FIXME: The following preprocessing transformations of the dataset are missing for 'seq'
#'input_types'       : ['img_crop','extrinsic_rot','extrinsic_rot_inv','bg_crop'],
#'output_types'      : ['3D','img_crop'],
#'label_types_train' : ['img_crop','3D','bounding_box_cam','intrinsic_crop','extrinsic_rot','extrinsic_rot_inv'],
#'label_types_test'  : ['img_crop','3D','bounding_box_cam','intrinsic_crop','extrinsic_rot','extrinsic_rot_inv'],

df['frame'] = df['imagefile'].apply(lambda x: x.split('.')[1].split('_')[1])
df = df.astype({'subj': int, 'cam': int, 'frame': int})

sample = df.iloc[1000]

plot_2d_pose(sample['imagefile'], sample['center'], sample['part'])

