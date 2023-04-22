import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers

import random
import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt


RANDOM_SEED = 42
batch_size = 512
epochs = 20
learning_rate = 1e-3

def set_seed():
    """Set seed for (1) tensorflow, (2) random and (3) numpy for stable results"""
    tf.random.set_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)


class DatasetMNIST():

    def __init__(self):
        (self.x_train, _), (self.x_test, _) = tf.keras.datasets.mnist.load_data()
        self.x_train = self.x_train / np.float32(255)
        self.x_test = self.x_test / np.float32(255)

    def sampling_generator(self, num_samples=1000, batch_size=8):
        for i in range(num_samples // batch_size):
            sample_ids = np.random.choice(self.x_train.shape[0], batch_size)
            sample = self.x_train[sample_ids]
            yield sample


if __name__ == "__main__":

    set_seed()

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / np.float32(255)
    x_test = x_test / np.float32(255)

    # Encoder architecture
    encoder = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu', name='input_to_hidden'),
        layers.Dense(128, activation='sigmoid', name='hidden_to_latent')
    ], name='encoder')

    encoder.summary()

    # Decoder architecture
    decoder = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(128, ), name='latent_to_hidden'),
        layers.Dense(784, activation='sigmoid', name='hidden_to_output'),
        layers.Reshape((28, 28))
    ], name='decoder')

    decoder.summary()

    # Autoencoder
    autoencoder = models.Model(inputs=encoder.input, outputs=decoder(encoder.output), name='autoencoder')
    autoencoder.summary()

    # Loss and optmizer
    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")

    # Train the model using compile/fit
#    autoencoder.compile(loss=loss, optimizer=optimizer)
#    autoencoder.fit(x_train, x_train, batch_size=batch_size, epochs=epochs, shuffle=False, verbose=2)

    # Train model using gradientTape
    dataset = DatasetMNIST()

    t = trange(0, desc='loss')
    for r in tqdm(range(0, epochs)):
        generator = dataset.sampling_generator(num_samples=60000, batch_size=batch_size)
        for batch_data in generator:
            X = batch_data
            Y = batch_data
            with tf.GradientTape() as tape:
                Y_hat = autoencoder(X, Y)
                total_loss = loss(X, Y_hat)
            gradients = tape.gradient(total_loss, autoencoder.trainable_variables)
            optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))
            total_loss_tracker.update_state(total_loss)
        t.set_description('total_loss=%g' % total_loss_tracker.result())

    # Test 10 samples and their reconstructed images
    sample_size = 10
    test_examples = x_test[:sample_size]

    reconstruction = autoencoder.predict(test_examples)

    plt.figure(figsize=(20, 4))
    for index in range(10):
        # display original
        ax = plt.subplot(2, sample_size, index + 1)
        plt.imshow(test_examples[index].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, sample_size, index + 1 + sample_size)
        plt.imshow(reconstruction[index].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

