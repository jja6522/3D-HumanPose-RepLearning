# 3D-HumanPose-RepL

This is a reproducibility study for: [DLow: Diversifying Latent FLows](https://arxiv.org/pdf/2003.08386.pdf)

## Setup

1. Install conda for your OS from https://conda.io/docs/user-guide/install/

2. Create an new environment via conda

- For Linux (CUDA 11.6 and Tensorflow 2.12.x)
    ```bash
    conda env create -f env_linux_tensorflow.yaml
    ```

## Datasets

This project was implemented using two datasets:

- [Human3.6m](http://vision.imar.ro/human3.6m/) was originally used in the paper and it was used to reproduce the baselines. The preprocessed 3D joint annotations can be downloaded from ([data_3d_h36m.npz](https://drive.google.com/file/d/1VrPFnUWxb56SXrkucy-HIxjcc6t80uxi/view?usp=share_link)) or follow the data preprocessing steps ([DATASETS.md](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md)) inside the [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) repo. Then, place the prepocessed data ``data_3d_h36m.npz`` under the ``data`` folder at the root of the project.

- [MPI-INF-3DHP](https://vcai.mpi-inf.mpg.de/3dhp-dataset/) was adapted and preprocessed accordingly to extract the 3D joint annotations. Then, it was used to train/test the DLow sampling and evaluate the results to advance the research using this model. The preprocessed 3D joint annotations can be downloaded from ...

## Train models

- Variational Autoencoder: Default 50 epochs
    ```python
    python src/train.py --model vae
    ```

- DLow Sampling: Default 50 epochs, 10 dlow_samples
    ```python
    python src/train.py --model dlow
    ```

## Test models quantitatively

- All models: Default loads all models (ae, vae, dlow) pre-trained for 50 epochs and 10 dlow_samples
    ```python
    python src/test.py
    ```

- Variational Autoencoder: Default loads model pre-trained for 50 epochs
    ```python
    python src/test.py --model vae
    ```

- DLow Sampling: Default loads model pre-trained for 50 epochs and 10 dlow_samples
    ```python
    python src/test.py --model dlow
    ```

## Evaluate models qualitatively

- Sampling: Default loads all models (ae, vae, dlow) pre-trained for 50 epochs and 10 dlow_samples

    ```python
    python src/eval.py --action sampling
    ```

- Reconstruction: Loads only autoencoder and VAE pre-trained for 50 epochs (DLow does not do reconstruction)

    ```python
    python src/eval.py --action reconstruct
    ```

