import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, Dense, UpSampling2D, Reshape

import random
import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

import os
from glob import glob

RANDOM_SEED = 42

batch_size = 512
epochs = 30
learning_rate = 1e-3
latent_dim = 50


def set_seed():
    """Set seed for (1) tensorflow, (2) random and (3) numpy for stable results"""
    tf.random.set_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)


def get_prior(num_modes, latent_dim):
    prior = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical(probs=[1 / num_modes,] * num_modes),
        components_distribution=tfp.distributions.MultivariateNormalDiag(
            loc=tf.Variable(tf.random.normal(shape=[num_modes, latent_dim])),
            scale_diag=tfp.util.TransformedVariable(tf.Variable(tf.ones(shape=[num_modes, latent_dim])), bijector=tfp.bijectors.Softplus())
        )
    )
    return prior


def get_kl_regularizer(prior_distribution):
    divergence_regularizer = tfp.layers.KLDivergenceRegularizer(
        prior_distribution,
        use_exact_kl=False,
        weight=1.0,
        test_points_fn=lambda q: q.sample(3),
        test_points_reduce_axis=(0, 1)
    )
    return divergence_regularizer


def reconstruction_loss(batch_of_images, decoding_dist):
    """
    The function takes batch_of_images (Tensor containing a batch of input images to
    the encoder) and decoding_dist (output distribution of decoder after passing the 
    image batch through the encoder and decoder) as arguments.
    The function should return the scalar average expected reconstruction loss.
    """
    return -tf.reduce_mean(decoding_dist.log_prob(batch_of_images), axis=0)


class VAE(keras.Model):
    """Variational autoencoder."""

    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.prior = get_prior(num_modes=2, latent_dim=latent_dim)
        self.kl_regularizer = get_kl_regularizer(self.prior)
        self.encoder = Sequential([
            Conv2D(32, (4, 4), activation='relu', strides=2, padding='SAME', input_shape=(64, 64, 3)),
            BatchNormalization(),
            Conv2D(64, (4, 4), activation='relu', strides=2, padding='SAME'),
            BatchNormalization(),
            Conv2D(128, (4, 4), activation='relu', strides=2, padding='SAME'),
            BatchNormalization(),
            Conv2D(256, (4, 4), activation='relu', strides=2, padding='SAME'),
            BatchNormalization(),
            Flatten(),
            Dense(tfp.layers.MultivariateNormalTriL.params_size(self.latent_dim)),
                  tfp.layers.MultivariateNormalTriL(self.latent_dim, activity_regularizer=self.kl_regularizer)
        ])

        self.decoder = Sequential([
            Dense(4096, activation='relu', input_shape=(self.latent_dim, )),
            Reshape((4, 4, 256)),
            UpSampling2D(size=(2, 2)),
            Conv2D(128, (3, 3), activation='relu', padding='SAME'),
            UpSampling2D(size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu', padding='SAME'),
            UpSampling2D(size=(2, 2)),
            Conv2D(32, (3, 3), activation='relu', padding='SAME'),
            UpSampling2D(size=(2, 2)),
            Conv2D(128, (3, 3), activation='relu', padding='SAME'),
            Conv2D(3, (3, 3), padding='SAME'),
            Flatten(),
            tfp.layers.IndependentBernoulli(event_shape=(64, 64, 3))
        ])

        self.model = Model(inputs=self.encoder.inputs, outputs=self.decoder(self.encoder.outputs))
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            z = self.encoder(x)
            y_rec = self.decoder(z)
            total_loss = reconstruction_loss(x, y_rec)
        gradients = tape.gradient(total_loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.total_loss_tracker.update_state(total_loss)
        return {
            "loss": self.total_loss_tracker.result()
        }

    def call(self, inputs):
        return self.model(inputs)


def get_encoder(latent_dim, kl_regularizer):
    encoder = Sequential([
        Conv2D(32, (4, 4), activation='relu', strides=2, padding='SAME', input_shape=(64, 64, 3)),
        BatchNormalization(),
        Conv2D(64, (4, 4), activation='relu', strides=2, padding='SAME'),
        BatchNormalization(),
        Conv2D(128, (4, 4), activation='relu', strides=2, padding='SAME'),
        BatchNormalization(),
        Conv2D(256, (4, 4), activation='relu', strides=2, padding='SAME'),
        BatchNormalization(),
        Flatten(),
        Dense(tfp.layers.MultivariateNormalTriL.params_size(latent_dim)),
              tfp.layers.MultivariateNormalTriL(latent_dim, activity_regularizer=kl_regularizer)
    ])
    return encoder


def get_decoder(latent_dim):
    decoder = Sequential([
        Dense(4096, activation='relu', input_shape=(latent_dim, )),
        Reshape((4, 4, 256)),
        UpSampling2D(size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='SAME'),
        UpSampling2D(size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='SAME'),
        UpSampling2D(size=(2, 2)),
        Conv2D(32, (3, 3), activation='relu', padding='SAME'),
        UpSampling2D(size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='SAME'),
        Conv2D(3, (3, 3), padding='SAME'),
        Flatten(),
        tfp.layers.IndependentBernoulli(event_shape=(64, 64, 3))
    ])
    return decoder


def reconstruct(encoder, decoder, batch_of_images):
    """
    The function takes the encoder, decoder and batch_of_images as inputs, which
    should be used to compute the reconstructions.
    The function should then return the reconstructions Tensor.
    """
    approx_posterior = encoder(batch_of_images)
    decoding_dist = decoder(approx_posterior.mean())
    return decoding_dist.mean()


def generate_images(prior, decoder, n_samples):
    """
    The function takes the prior distribution, decoder and number of samples as inputs, which
    should be used to generate the images.
    The function should then return the batch of generated images.
    """
    z = prior.sample(n_samples)
    return decoder(z).mean()


if __name__ == "__main__":

    set_seed()

    #######################################
    # Dataset Preparation
    #######################################
    train_ds = tf.keras.utils.image_dataset_from_directory(
      '/home/johann/datasets/CelebA/img_align_celeba',
      labels=None,
      validation_split=0.2,
      subset="training",
      seed=RANDOM_SEED,
      image_size=(64, 64),
      batch_size=None)

    val_ds = tf.keras.utils.image_dataset_from_directory(
      '/home/johann/datasets/CelebA/img_align_celeba',
      labels=None,
      validation_split=0.2,
      subset="validation",
      seed=RANDOM_SEED,
      image_size=(64, 64),
      batch_size=None)

    test_ds = tf.keras.utils.image_dataset_from_directory(
      '/home/johann/datasets/CelebA/img_align_celeba',
      labels=None,
      seed=RANDOM_SEED,
      image_size=(64, 64),
      batch_size=None)

    #######################################
    # Dataset testing
    #######################################
    n_examples_shown = 6
    f, axs = plt.subplots(1, n_examples_shown, figsize=(16, 3))
    for j, image in enumerate(train_ds.take(n_examples_shown)):
        print(image.shape)
        axs[j].imshow(image.numpy().astype("uint8"))
        axs[j].axis('off')
    plt.show()

    # Reduce the dataset to 10000 to training and batch it
    train_ds = train_ds.take(10000)
    train_ds = train_ds.batch(32)
    train_ds = train_ds.map(lambda x: x / 255.0)
    train_ds = train_ds.map(lambda x: (x, x))

    # Reduce the dataset to 1000 for validation and batch it
    val_ds = val_ds.take(1000)
    val_ds = val_ds.batch(32)
    val_ds = val_ds.map(lambda x: x / 255.0)
    val_ds = val_ds.map(lambda x: (x, x))

    # Reduce the dataset to 100 for test and batch it
    test_ds = test_ds.take(100)
    test_ds = test_ds.batch(32)
    test_ds = test_ds.map(lambda x: x / 255.0)
    test_ds = test_ds.map(lambda x: (x, x))

    #######################################
    # Variational Autoencoder architecture
    #######################################
    # Define the prior distribution
    #prior = get_prior(num_modes=2, latent_dim=50)
    #kl_regularizer = get_kl_regularizer(prior)

    # Define the encoder Network
    #encoder = get_encoder(latent_dim=50, kl_regularizer=kl_regularizer)
    #encoder.summary()

    # Define the decoder network
    #decoder = get_decoder(latent_dim=50)
    #decoder.summary()

    # Link the encoder and decoder together
    #vae = Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs))
    vae = VAE(latent_dim)

    # Compile and fit the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    # Train the model using compile/fit
    vae.compile(optimizer=optimizer, loss=reconstruction_loss)
    vae.fit(train_ds, validation_data=val_ds, epochs=30, verbose=2)

    #######################################
    # Reconstruction examples
    #######################################
    n_reconstructions = 7
    test_ds_for_reconstructions = tf.keras.utils.image_dataset_from_directory(
      '/home/johann/datasets/CelebA/img_align_celeba',
      labels=None,
      seed=RANDOM_SEED,
      image_size=(64, 64),
      batch_size=None)

    test_ds_for_reconstructions = test_ds_for_reconstructions.take(100).shuffle(100)
    test_ds_for_reconstructions = test_ds_for_reconstructions.map(lambda x: x / 255.0)

    example_images = None
    for all_test_images in test_ds_for_reconstructions.shuffle(100).batch(n_reconstructions).take(1):
        example_images = all_test_images.numpy()

    reconstructions = reconstruct(vae.encoder, vae.decoder, example_images).numpy()

    # Plot the reconstructions
    f, axs = plt.subplots(2, n_reconstructions, figsize=(16, 6))
    axs[0, n_reconstructions // 2].set_title("Original test images")
    axs[1, n_reconstructions // 2].set_title("Reconstructed images")
    for j in range(n_reconstructions):
        axs[0, j].imshow(example_images[j])
        axs[1, j].imshow(reconstructions[j])
        axs[0, j].axis('off')
        axs[1, j].axis('off')
        
    plt.tight_layout()
    plt.show()

    ######################################
    # New Image Generation Examples
    ######################################
    n_samples = 10
    sampled_images = generate_images(vae.prior, vae.decoder, n_samples)

    f, axs = plt.subplots(1, n_samples, figsize=(16, 6))

    for j in range(n_samples):
        axs[j].imshow(sampled_images[j])
        axs[j].axis('off')
        
    plt.tight_layout()
    plt.show()

