# 3D-HumanPose-RepLearning

This is a reproducibility study for the paper on Unsupervised Geometry-Aware Representation for 3D Human Pose Estimation at https://arxiv.org/abs/1804.01110

## Setup

1. Install conda for your OS from https://conda.io/docs/user-guide/install/

2. Create an new environment via conda

- For Linux (CUDA 11.6)
    ```bash
    conda env create -f env_linux.yaml
    ```

3. Download the dataset from http://vision.imar.ro/ or alternatively use the links below:

- Annotations (~350MB):
    ```bash
    wget http://visiondata.cis.upenn.edu/volumetric/h36m/h36m_annot.tar

    ```
- Images (~26GB):
    ```bash
    wget http://visiondata.cis.upenn.edu/volumetric/h36m/S1.tar
    wget http://visiondata.cis.upenn.edu/volumetric/h36m/S5.tar
    wget http://visiondata.cis.upenn.edu/volumetric/h36m/S6.tar
    wget http://visiondata.cis.upenn.edu/volumetric/h36m/S7.tar
    wget http://visiondata.cis.upenn.edu/volumetric/h36m/S8.tar
    wget http://visiondata.cis.upenn.edu/volumetric/h36m/S9.tar
    wget http://visiondata.cis.upenn.edu/volumetric/h36m/S11.tar

    ```
