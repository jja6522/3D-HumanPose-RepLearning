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

        # Encoder RNN for conditional past motions c
        self.enc_rnn_c = models.Sequential([
            layers.GRU(units=nh_rnn, input_shape=(t_his, traj_dim))
        ], name="enc_rnn_c")

        # Encoder RNN for predicted future motions x
        self.enc_rnn_x = models.Sequential([
            layers.GRU(units=nh_rnn, input_shape=(t_pred, traj_dim))
        ], name="enc_rnn_x")

        # Encoder MLP for both past and future motions
        self.enc_mlp = models.Sequential([
            layers.InputLayer(input_shape=(2 * nh_rnn)),
            layers.Dense(300, activation='tanh', name='enc_mlp1'),
            layers.Dense(200, activation='tanh', name='enc_mlp2'),
            # Simple latent dimension
            layers.Dense(latent_dim, activation=None, name='latent')
        ], name="enc_mlp")

        # Decoder MLP for both past and future motions
        self.dec_mlp = models.Sequential([
            layers.InputLayer(input_shape=(t_pred, latent_dim)),
            layers.Dense(200, activation='tanh', name='dec_mlp2'),
            layers.Dense(300, activation='tanh', name='dec_mlp1'),
            layers.Dense(traj_dim)
        ], name="dec_mlp")

        # Decoder RNN for predicted future motions x
        self.dec_rnn_x = models.Sequential([
            layers.InputLayer(input_shape=(latent_dim + nh_rnn)),
            layers.RepeatVector(t_pred),
            layers.GRU(units=nh_rnn, input_shape=(t_pred, traj_dim), return_sequences=True),
        ], name="dec_rnn_x")

    def encode(self, x, c):
        h_x = self.enc_rnn_x(x)
        h_c = self.enc_rnn_c(c)
        h = tf.concat((h_x, h_c), axis=1)
        z = self.enc_mlp(h)
        return z

    def decode(self, z, c):
        h_c = self.enc_rnn_c(c)
        h = tf.concat((h_c, z), axis=1)
        h_x = self.dec_rnn_x(h)
        x_rec = self.dec_mlp(h_x)
        return x_rec

    @tf.function
    def sample_prior(self, c):
        z_sample = tf.random.uniform(shape=[c.shape[0], self.latent_dim])
        x_rec = self.decode(z_sample, c)
        return x_rec

    def summary(self):
        self.enc_rnn_c.summary()
        self.enc_rnn_x.summary()
        self.enc_mlp.summary()
        self.dec_mlp.summary()
        self.dec_rnn_x.summary()

    def save_model(self, num_epochs):
        self.enc_rnn_c.save(f"models/ae-enc_rnn_c-{num_epochs}.model")
        self.enc_rnn_x.save(f"models/ae-enc_rnn_x-{num_epochs}.model")
        self.enc_mlp.save(f"models/ae-enc_mlp-{num_epochs}.model")
        self.dec_mlp.save(f"models/ae-dec_mlp-{num_epochs}.model")
        self.dec_rnn_x.save(f"models/ae-dec_rnn_x-{num_epochs}.model")

    def load_model(self, num_epochs):
        self.enc_rnn_c = models.load_model(f"models/ae-enc_rnn_c-{num_epochs}.model")
        self.enc_rnn_x = models.load_model(f"models/ae-enc_rnn_x-{num_epochs}.model")
        self.enc_mlp = models.load_model(f"models/ae-enc_mlp-{num_epochs}.model")
        self.dec_mlp = models.load_model(f"models/ae-dec_mlp-{num_epochs}.model")
        self.dec_rnn_x = models.load_model(f"models/ae-dec_rnn_x-{num_epochs}.model")


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

        # Encoder RNN for conditional past motions c
        self.enc_rnn_c = models.Sequential([
            layers.GRU(units=nh_rnn, input_shape=(t_his, traj_dim))
        ], name="enc_rnn_c")

        # Encoder RNN for predicted future motions x
        self.enc_rnn_x = models.Sequential([
            layers.GRU(units=nh_rnn, input_shape=(t_pred, traj_dim))
        ], name="enc_rnn_x")

        # Encoder MLP for both past and future motions
        self.enc_mlp = models.Sequential([
            layers.InputLayer(input_shape=(2 * nh_rnn)),
            layers.Dense(300, activation='tanh', name='enc_mlp1'),
            layers.Dense(200, activation='tanh', name='enc_mlp2'),
            # latent layer for mu and sigma
            layers.Dense(latent_dim + latent_dim, activation=None, name='latent')
        ], name="enc_mlp")

        # Decoder MLP for both past and future motions
        self.dec_mlp = models.Sequential([
            layers.InputLayer(input_shape=(t_pred, latent_dim)),
            layers.Dense(200, activation='tanh', name='dec_mlp2'),
            layers.Dense(300, activation='tanh', name='dec_mlp1'),
            layers.Dense(traj_dim)
        ], name="dec_mlp")

        # Decoder RNN for predicted future motions x
        self.dec_rnn_x = models.Sequential([
            layers.InputLayer(input_shape=(latent_dim + nh_rnn)),
            layers.RepeatVector(t_pred),
            layers.GRU(units=nh_rnn, input_shape=(t_pred, traj_dim), return_sequences=True),
        ], name="dec_rnn_x")

    def encode(self, x, c):
        h_x = self.enc_rnn_x(x)
        h_c = self.enc_rnn_c(c)
        h = tf.concat((h_x, h_c), axis=1)
        z_dist = self.enc_mlp(h)
        mu, logvar = tf.split(z_dist, num_or_size_splits=2, axis=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = tf.exp(0.5 * logvar)
        eps = tf.random.normal(shape=mu.shape)
        return mu + eps * std

    def decode(self, z, c):
        h_c = self.enc_rnn_c(c)
        h = tf.concat((h_c, z), axis=1)
        h_x = self.dec_rnn_x(h)
        x_rec = self.dec_mlp(h_x)
        return x_rec

    @tf.function
    def sample_prior(self, c):
        z_sample = tf.random.normal(shape=[c.shape[0], self.latent_dim])
        x_rec = self.decode(z_sample, c)
        return x_rec

    def summary(self):
        self.enc_rnn_c.summary()
        self.enc_rnn_x.summary()
        self.enc_mlp.summary()
        self.dec_mlp.summary()
        self.dec_rnn_x.summary()

    def save_model(self, num_epochs):
        self.enc_rnn_c.save(f"models/vae-enc_rnn_c-{num_epochs}.model")
        self.enc_rnn_x.save(f"models/vae-enc_rnn_x-{num_epochs}.model")
        self.enc_mlp.save(f"models/vae-enc_mlp-{num_epochs}.model")
        self.dec_mlp.save(f"models/vae-dec_mlp-{num_epochs}.model")
        self.dec_rnn_x.save(f"models/vae-dec_rnn_x-{num_epochs}.model")

    def load_model(self, num_epochs):
        self.enc_rnn_c = models.load_model(f"models/vae-enc_rnn_c-{num_epochs}.model")
        self.enc_rnn_x = models.load_model(f"models/vae-enc_rnn_x-{num_epochs}.model")
        self.enc_mlp = models.load_model(f"models/vae-enc_mlp-{num_epochs}.model")
        self.dec_mlp = models.load_model(f"models/vae-dec_mlp-{num_epochs}.model")
        self.dec_rnn_x = models.load_model(f"models/vae-dec_rnn_x-{num_epochs}.model")


class DLow(keras.Model):
    """DLow Encoder."""

    def __init__(
        self,
        traj_dim=48,
        t_his=25,
        t_pred=100,
        latent_dim=128,
        nh_rnn = 128,
        dlow_samples = 10,
        name="dlow",
        **kwargs
    ):
        super(DLow, self).__init__(name=name, **kwargs)
        #TODO: This attributes are not restored when loading a pre-trained model
        self.t_his = t_his
        self.t_pred = t_pred
        self.latent_dim = latent_dim
        self.traj_dim = traj_dim
        self.dlow_samples = dlow_samples

        # Encoder RNN for conditional past motions c
        self.enc_rnn_c = models.Sequential([
            layers.GRU(units=nh_rnn, input_shape=(t_his, traj_dim))
        ], name="enc_dlow_c")

        # Encoder MLP for latent mapping functions parameters
        self.enc_mlp = models.Sequential([
            layers.InputLayer(input_shape=(nh_rnn)),
            layers.Dense(1024, activation='tanh', name='enc_mlp1'),
            layers.Dense(512, activation='tanh', name='enc_mlp2')
        ], name="enc_dlow_mlp")

        # Layers for affine transformations for the k dlow_samples
        self.rotation_scaling = models.Sequential([
            layers.InputLayer(input_shape=(512)),
            layers.Dense(latent_dim * dlow_samples, activation='tanh'),
            layers.Reshape([dlow_samples, latent_dim])
        ], name="rotation_scaling")

        self.translation = models.Sequential([
            layers.InputLayer(input_shape=(512)),
            layers.Dense(latent_dim * dlow_samples, activation='tanh'),
            layers.Reshape([dlow_samples, latent_dim])
        ], name="translation")

    def encode(self, c):

        # Use the conditioned past motion c to get the affine transformation parameters
        h_c = self.enc_rnn_c(c)
        h = self.enc_mlp(h_c)
        a = self.rotation_scaling(h)
        b = self.translation(h)

        # Common gaussian variable epsilon to apply the affine transformations
        epsilon = tf.random.normal(shape=[c.shape[0], 1, self.latent_dim])
        #epsilon = tf.random.normal(shape=[h_c.shape[0], self.dlow_samples, self.latent_dim])

        # Repeat the same epsilon for the dlow_samples for training/testing
        epsilon_rep = tf.repeat(epsilon, repeats = [self.dlow_samples], axis=1)

        # Apply the affine transformation for rotation, scaling and translation
        z = a * epsilon_rep + b

        # Return latent codes zk and affine mapping functions
        return z, a, b

    @tf.function
    def sample_prior(self, c):
        z_new, _, _ = self.encode(c)
        return z_new

    def summary(self):
        self.enc_rnn_c.summary()
        self.enc_mlp.summary()
        self.rotation_scaling.summary()
        self.translation.summary()

    def save_model(self, num_epochs, dlow_samples):
        self.enc_rnn_c.save(f"models/dlow{dlow_samples}-enc_rnn_c-{num_epochs}.model")
        self.enc_mlp.save(f"models/dlow{dlow_samples}-enc_mlp-{num_epochs}.model")
        self.rotation_scaling.save(f"models/dlow{dlow_samples}-rotation_scaling-{num_epochs}.model")
        self.translation.save(f"models/dlow{dlow_samples}-translation-{num_epochs}.model")

    def load_model(self, num_epochs, dlow_samples):
        self.dlow_samples = dlow_samples
        self.enc_rnn_c = models.load_model(f"models/dlow{dlow_samples}-enc_rnn_c-{num_epochs}.model")
        self.enc_mlp = models.load_model(f"models/dlow{dlow_samples}-enc_mlp-{num_epochs}.model")
        self.rotation_scaling = models.load_model(f"models/dlow{dlow_samples}-rotation_scaling-{num_epochs}.model")
        self.translation = models.load_model(f"models/dlow{dlow_samples}-translation-{num_epochs}.model")

