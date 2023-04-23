import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers


class AE(keras.Model):
    """Simple Autoencoder."""

    def __init__(
        self,
        traj_dim=48,
        t_his=25,
        t_pred=100,
        latent_dim=200,
        nh_rnn = 128,
        name="autoencoder",
        **kwargs
    ):
        super(AE, self).__init__(name=name, **kwargs)
        self.latent_dim = latent_dim

        # Encoder architecture
        self.encoder = models.Sequential([
            layers.GRU(units=nh_rnn, input_shape=(t_his, traj_dim)),
            layers.Dense(300, activation='tanh', name='enc_mlp1'),
            layers.Dense(latent_dim, activation='tanh', name='enc_mlp2')
        ], name="encoder")

        # Decoder architecture
        self.decoder = models.Sequential([
            layers.InputLayer(latent_dim),
            layers.RepeatVector(t_pred),
            layers.GRU(units=nh_rnn, return_sequences=True),
            layers.Dense(300, activation='tanh', name='dec_mlp1'),
            layers.Dense(200, activation='tanh', name='dec_mlp2'),
            layers.Dense(traj_dim, activation='tanh', name='dec_mlp3')
        ], name="decoder")

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


class VAE(keras.Model):
    """Variational autoencoder."""

    def __init__(
        self,
        traj_dim=48,
        t_his=25,
        t_pred=100,
        latent_dim=200,
        nh_rnn = 128,
        name="vae",
        **kwargs
    ):
        super(VAE, self).__init__(name=name, **kwargs)
        self.latent_dim = latent_dim

        # Encoder architecture
        self.encoder = models.Sequential([
            layers.GRU(units=nh_rnn, input_shape=(t_his, traj_dim)),
            layers.Dense(300, activation='tanh', name='enc_mlp1'),
            layers.Dense(200, activation='tanh', name='enc_mlp2'),
            layers.Dense(latent_dim + latent_dim, activation=None, name='latent')
        ], name="encoder")

        # Decoder architecture
        self.decoder = models.Sequential([
            layers.InputLayer(latent_dim),
            layers.RepeatVector(t_pred),
            layers.GRU(units=nh_rnn, return_sequences=True),
            layers.Dense(300, activation='tanh', name='dec_mlp1'),
            layers.Dense(200, activation='tanh', name='dec_mlp2'),
            layers.Dense(traj_dim, activation=None, name='dec_mlp3')
        ], name="decoder")

    def encode(self, x):
        mu, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = tf.exp(0.5 * logvar)
        eps = tf.random.normal(shape=mu.shape)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

