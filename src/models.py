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
        latent_dim=128,
        nh_rnn = 128,
        name="ae",
        **kwargs
    ):
        super(AE, self).__init__(name=name, **kwargs)
        self.t_his = t_his
        self.t_pred = t_pred
        self.latent_dim = latent_dim
        self.traj_dim = traj_dim

        # Encoder RNN for past motions c
        self.enc_rnn_past = models.Sequential([
            layers.GRU(units=nh_rnn, input_shape=(t_his, traj_dim))
        ], name="enc_rnn_past")

        # Encoder RNN for future motions x
        self.enc_rnn_future = models.Sequential([
            layers.GRU(units=nh_rnn, input_shape=(t_pred, traj_dim))
        ], name="enc_rnn_future")

        # Encoder MLP for both past and future motions
        self.enc_mlp = models.Sequential([
            layers.InputLayer(input_shape=(2 * nh_rnn)),
            layers.Dense(300, activation='tanh', name='enc_mlp1'),
            layers.Dense(200, activation='tanh', name='enc_mlp2'),
            # Simple latent layer
            layers.Dense(latent_dim, activation=None, name='latent')
        ], name="enc_mlp")

        # Decoder MLP for both past and future motions
        self.dec_mlp = models.Sequential([
            layers.InputLayer(latent_dim),
            layers.Dense(200, activation='tanh', name='dec_mlp2'),
            layers.Dense(300, activation='tanh', name='dec_mlp1'),
        ], name="dec_mlp")

        # Decoder RNN for future motions x
        self.dec_rnn_future = models.Sequential([
            layers.InputLayer(input_shape=(300)),
            layers.RepeatVector(t_pred),
            layers.GRU(units=nh_rnn, input_shape=(t_pred, traj_dim), return_sequences=True),
            layers.Dense(traj_dim)
        ], name="dec_rnn_future")

    def encode(self, x, y):
        h_x = self.enc_rnn_past(x)
        h_y = self.enc_rnn_future(y)
        h = tf.concat((h_x, h_y), axis=1)
        z = self.enc_mlp(h)
        return z

    def decode(self, z):
        h_y = self.dec_mlp(z)
        y_rec = self.dec_rnn_future(h_y)
        return y_rec

    # FIXME: If there a better way to sample from the latent space?
    def sample(self, x):
        z_sample = tf.random.uniform(shape=[x.shape[0], self.latent_dim])
        y = self.decode(z_sample)
        z = self.encode(x, y)
        y_new = self.decode(z)
        return y_new

    def summary(self):
        self.enc_rnn_past.summary()
        self.enc_rnn_future.summary()
        self.enc_mlp.summary()
        self.dec_mlp.summary()
        self.dec_rnn_future.summary()

    def save_model(self, num_epochs):
        self.enc_rnn_past.save(f"models/ae-enc_rnn_past-{num_epochs}.model")
        self.enc_rnn_future.save(f"models/ae-enc_rnn_future-{num_epochs}.model")
        self.enc_mlp.save(f"models/ae-enc_mlp-{num_epochs}.model")
        self.dec_mlp.save(f"models/ae-dec_mlp-{num_epochs}.model")
        self.dec_rnn_future.save(f"models/ae-dec_rnn_future-{num_epochs}.model")

    def load_model(self, num_epochs):
        self.enc_rnn_past = models.load_model(f"models/ae-enc_rnn_past-{num_epochs}.model")
        self.enc_rnn_future = models.load_model(f"models/ae-enc_rnn_future-{num_epochs}.model")
        self.enc_mlp = models.load_model(f"models/ae-enc_mlp-{num_epochs}.model")
        self.dec_mlp = models.load_model(f"models/ae-dec_mlp-{num_epochs}.model")
        self.dec_rnn_future = models.load_model(f"models/ae-dec_rnn_future-{num_epochs}.model")


class VAE(keras.Model):
    """Variational autoencoder."""

    def __init__(
        self,
        traj_dim=48,
        t_his=25,
        t_pred=100,
        latent_dim=128,
        nh_rnn = 128,
        name="vae",
        **kwargs
    ):
        super(VAE, self).__init__(name=name, **kwargs)
        self.t_his = t_his
        self.t_pred = t_pred
        self.latent_dim = latent_dim
        self.traj_dim = traj_dim

        # Encoder RNN for past motions c
        self.enc_rnn_past = models.Sequential([
            layers.GRU(units=nh_rnn, input_shape=(t_his, traj_dim))
        ], name="enc_rnn_past")

        # Encoder RNN for future motions x
        self.enc_rnn_future = models.Sequential([
            layers.GRU(units=nh_rnn, input_shape=(t_pred, traj_dim))
        ], name="enc_rnn_future")

        # Encoder MLP for both past and future motions
        self.enc_mlp = models.Sequential([
            layers.InputLayer(input_shape=(2 * nh_rnn)),
            layers.Dense(300, activation='tanh', name='enc_mlp1'),
            layers.Dense(200, activation='tanh', name='enc_mlp2'),
            # FIXME: Double check if a multivariate gaussian per joint is required
            # latent layer for mu and sigma
            layers.Dense(latent_dim + latent_dim, activation=None, name='latent')
        ], name="enc_mlp")

        # Decoder MLP for both past and future motions
        self.dec_mlp = models.Sequential([
            layers.InputLayer(latent_dim),
            layers.Dense(200, activation='tanh', name='dec_mlp2'),
            layers.Dense(300, activation='tanh', name='dec_mlp1'),
        ], name="dec_mlp")

        # Decoder RNN for future motions x
        self.dec_rnn_future = models.Sequential([
            layers.InputLayer(input_shape=(300)),
            layers.RepeatVector(t_pred),
            layers.GRU(units=nh_rnn, input_shape=(t_pred, traj_dim), return_sequences=True),
            layers.Dense(traj_dim)
        ], name="dec_rnn_future")

    def encode(self, x, y):
        h_x = self.enc_rnn_past(x)
        h_y = self.enc_rnn_future(y)
        h = tf.concat((h_x, h_y), axis=1)
        mu, logvar = tf.split(self.enc_mlp(h), num_or_size_splits=2, axis=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = tf.exp(0.5 * logvar)
        eps = tf.random.normal(shape=mu.shape)
        return mu + eps * std

    def decode(self, z):
        h_y = self.dec_mlp(z)
        y_rec = self.dec_rnn_future(h_y)
        return y_rec

    def sample(self, x):
        z_sample = tf.random.normal(shape=[x.shape[0], self.latent_dim])
        y = self.decode(z_sample)
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        y_new = self.decode(z)
        return y_new

    def summary(self):
        self.enc_rnn_past.summary()
        self.enc_rnn_future.summary()
        self.enc_mlp.summary()
        self.dec_mlp.summary()
        self.dec_rnn_future.summary()

    def save_model(self, num_epochs):
        self.enc_rnn_past.save(f"models/vae-enc_rnn_past-{num_epochs}.model")
        self.enc_rnn_future.save(f"models/vae-enc_rnn_future-{num_epochs}.model")
        self.enc_mlp.save(f"models/vae-enc_mlp-{num_epochs}.model")
        self.dec_mlp.save(f"models/vae-dec_mlp-{num_epochs}.model")
        self.dec_rnn_future.save(f"models/vae-dec_rnn_future-{num_epochs}.model")

    def load_model(self, num_epochs):
        self.enc_rnn_past = models.load_model(f"models/vae-enc_rnn_past-{num_epochs}.model")
        self.enc_rnn_future = models.load_model(f"models/vae-enc_rnn_future-{num_epochs}.model")
        self.enc_mlp = models.load_model(f"models/vae-enc_mlp-{num_epochs}.model")
        self.dec_mlp = models.load_model(f"models/vae-dec_mlp-{num_epochs}.model")
        self.dec_rnn_future = models.load_model(f"models/vae-dec_rnn_future-{num_epochs}.model")

