import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Reshape

import random
import numpy as np
import matplotlib.pyplot as plt


RANDOM_SEED = 42
RAND_INIT = tf.random_normal_initializer(stddev=0.1)

batch_size = 512
epochs = 20
learning_rate = 1e-3

def set_seed():
    """Set seed for (1) tensorflow, (2) random and (3) numpy for stable results"""
    tf.random.set_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)


if __name__ == "__main__":

    set_seed()

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / np.float32(255)
    x_test = x_test / np.float32(255)

    # Encoder architecture
    encoder = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu', name='input_to_hidden'),
        Dense(128, activation='sigmoid', name='hidden_to_latent')
    ], name='encoder')

    encoder.summary()

    # Decoder architecture
    decoder = Sequential([
        Dense(128, activation='relu', input_shape=(128, ), name='latent_to_hidden'),
        Dense(784, activation='sigmoid', name='hidden_to_output'),
        Reshape((28, 28))
    ], name='decoder')

    decoder.summary()

    # Autoencoder
    autoencoder = Model(inputs=encoder.input, outputs=decoder(encoder.output), name='autoencoder')
    autoencoder.summary()

    # Loss and optmizer
    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    autoencoder.compile(loss=loss, optimizer=optimizer)

    # Train the model
    autoencoder.fit(x_train, x_train, batch_size=batch_size, epochs=epochs, shuffle=False, verbose=2)

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

