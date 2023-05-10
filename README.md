# CSCI 736 NN&ML (Neural Computation): Research Project

This project is a reproducibility study for: [DLow: Diversifying Latent FLows](https://arxiv.org/pdf/2003.08386.pdf)

## Setup

1. Install conda for your OS from https://conda.io/docs/user-guide/install/

2. Create an new environment via conda

- For Linux (CUDA 11.6 and Tensorflow 2.12.x)
    ```bash
    conda env create -f env_linux_tensorflow.yaml
    ```

## Datasets

This project was implemented using two datasets:

- [Human3.6m](http://vision.imar.ro/human3.6m/) was originally used in the paper and it was used to reproduce the baselines. The preprocessed 3D joint annotations can be downloaded from ([data_3d_h36m.npz](https://drive.google.com/file/d/1VrPFnUWxb56SXrkucy-HIxjcc6t80uxi/view?usp=share_link)) or follow the data preprocessing steps ([DATASETS.md](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md)) inside the [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) repo.

- [MPI-INF-3DHP](https://vcai.mpi-inf.mpg.de/3dhp-dataset/) was adapted and preprocessed accordingly to extract the 3D joint annotations. Then, it was used to train/test the DLow sampling and evaluate the results to advance the research using this model. The preprocessed 3D joint annotations can be downloaded from ([data_3dhp.npz](https://drive.google.com/file/d/1fJvPXi1-jTBJ4EV_1HNgpNTD-NOM4cM3/view?usp=share_link)) or alternatively you can download the full dataset from their website, and use our preprocessing tool by runnning ``python src/preprocess_3dhp.py --data_path /path/to/downloaded/dataset``

*Note:*: In order to train the models and use any of these datasets, a preprocessed version is required to be placed into the ``data`` folder under the root directory of the project.

## Train

1. Train the CVAE first:
    ```python
    python src/train.py --model vae --dataset h36m --num_epochs 500 --dlow_samples 50
    ```

2. Train DLow after the CVAE is trained:
    ```python
    python src/train.py --model dlow --dataset h36m --num_epochs 500 --dlow_samples 50 --batch_size 32
    ```

- For a detailed usage of parameters see below:
    ```
    usage: train.py [-h] [--model MODEL] [--dataset DATASET] [--num_epochs NUM_EPOCHS] [--samples_per_epoch SAMPLES_PER_EPOCH] [--batch_size BATCH_SIZE] [--dlow_samples DLOW_SAMPLES]

    optional arguments:
      -h, --help            show this help message and exit
      --model MODEL         ae, vae, dlow
      --dataset DATASET     h36m, 3dhp
      --num_epochs NUM_EPOCHS
                            Number of epochs for training
      --samples_per_epoch SAMPLES_PER_EPOCH
                            Number of samples per epoch
      --batch_size BATCH_SIZE
                            Batch size
      --dlow_samples DLOW_SAMPLES
                            Number of DLow samples for epsilon (nk)
    ```

*Note:* The pre-trained models for both CVAE and DLow in our implementation can be downloaded from this link ([models.zip](https://drive.google.com/file/d/1Ur96Byf7JJIISB-gWxvhjkQX3TNTAQN2/view?usp=share_link)). Unzip the file in the root directory of the project and you can proceed to test/eval

## Test

- Test a pre-trained DLow model and collect statistics for diversity/accuracy
    ```python
    python src/test.py --model dlow --dataset h36m --num_epochs 500 --dlow_samples 50
    ```

- For a detailed usage of parameters see below:
    ```python
    usage: test.py [-h] [--model MODEL] [--dataset DATASET] [--num_epochs NUM_EPOCHS] [--dlow_samples DLOW_SAMPLES] [--num_seeds NUM_SEEDS] [--multimodal_threshold MULTIMODAL_THRESHOLD]

    optional arguments:
      -h, --help            show this help message and exit
      --model MODEL         all, ae, vae, dlow
      --dataset DATASET     h36m, 3dhp
      --num_epochs NUM_EPOCHS
                            Number of epochs to load a model
      --dlow_samples DLOW_SAMPLES
                            Number of DLow samples for epsilon (nk)
      --num_seeds NUM_SEEDS
      --multimodal_threshold MULTIMODAL_THRESHOLD
    ```

## Evaluate the models (visually)

- This command will launch an animation to show DLow Sampling diversity
```python
python src/eval.py --model dlow --dataset h36m --num_epochs 500 --dlow_samples 50
```

- For a detailed usage of parameters see below:
    ```python
    usage: eval.py [-h] [--model MODEL] [--dataset DATASET] [--num_epochs NUM_EPOCHS] [--dlow_samples DLOW_SAMPLES] [--vis_samples VIS_SAMPLES] [--action ACTION]

    optional arguments:
      -h, --help            show this help message and exit
      --model MODEL         all, ae, vae, dlow
      --dataset DATASET     h36m, 3dhp
      --num_epochs NUM_EPOCHS
                            Number of epochs to load a model
      --dlow_samples DLOW_SAMPLES
                            Number of DLow samples for epsilon (nk)
      --vis_samples VIS_SAMPLES
                            Number of samples to visualize
      --action ACTION       reconstruct, sampling
    ```

