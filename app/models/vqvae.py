from keras import layers as Layer, Input, Model
from keras.metrics import Mean
import tensorflow as tf


class ResidualStack(tf.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 name=None):
        super(ResidualStack, self).__init__(name=name)
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self._layers = []
        for idx in range(num_residual_layers):
            conv3 = Layer.Conv2D(num_residual_hiddens, kernel_size=3, strides=1, padding='same', name=f'res3x3_{idx}')
            bnorm1 = tf.keras.layers.BatchNormalization()
            conv1 = Layer.Conv2D(num_hiddens, kernel_size=1, strides=1, padding='same', name=f'res1x1_{idx}')
            self._layers.append((conv3, bnorm1, conv1))

    def __call__(self, inputs):
        h = inputs
        for conv3, bnorm1, conv1 in self._layers:
            conv3_out_l = conv3(tf.nn.relu(h))
            bnorm_out_l = bnorm1(conv3_out_l)
            conv1_out_l = conv1(tf.nn.relu(bnorm_out_l))
            h += conv1_out_l
        return tf.nn.relu(h)


class Encoder(Model):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, latent_dim,
                 name=None):
        super(Encoder, self).__init__(name=name)

        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens
        self._latent_dim = latent_dim

        self._enc_l1 = Layer.Conv2D(self._num_hiddens // 2, kernel_size=(4, 4), strides=(2, 2), padding='same',
                                    name='enc_l1')
        self._enc_l2 = Layer.Conv2D(self._num_hiddens, kernel_size=(4, 4), strides=(2, 2), padding='same',
                                    name='enc_l2')
        self._enc_l3 = Layer.Conv2D(self._num_hiddens, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                    name='enc_l3')
        self._residual_stack = ResidualStack(self._num_hiddens, self._num_residual_layers, self._num_residual_hiddens,
                                             name='resblock1')

        self._b_l1 = tf.keras.layers.BatchNormalization()
        self._enc_l4 = Layer.Conv2D(latent_dim, kernel_size=1, strides=1, padding='same', name=f'enc_l4')

    def call(self, input, training=None, mask=None):
        h = tf.nn.relu(self._enc_l1(input))
        h = tf.nn.relu(self._enc_l2(h))
        h = tf.nn.relu(self._enc_l3(h))
        h = self._residual_stack(h)
        h = self._b_l1(h)

        return tf.nn.relu(self._enc_l4(h))


class VectorQuantizer(Layer.Layer):
    # beta -> commitment cost
    def __init__(self, embedding_dim, num_embeddings, beta=0.25, name=None):
        super(VectorQuantizer, self).__init__(name=name)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta

        # Initialize the embeddings to quantize by pre-specifying random uniform distribution
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, inputs, training=None, mask=None):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact
        input_shape = tf.shape(inputs)
        flattened = tf.reshape(inputs, [-1, self.embedding_dim])

        # Quantization
        encoding_indices = self.get_code_indices(flattened)
        # Apply one hot
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        # Compute Matrix Multiplication
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)

        # Reshape the quantized values back to the original input shape
        quantized = tf.reshape(quantized, input_shape)

        # Calculate the vector quantization loss
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(inputs)) ** 2)
        #  Add the calculation to the layer
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        # Straight-through estimator
        quantized = inputs + tf.stop_gradient(quantized - inputs)
        return quantized

    def get_code_indices(self, flattened_inputs):
        # Calculate the L2-normalized distance between the inputs and the codes
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        # Obtain distance distribution
        distances = (
                tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
                + tf.reduce_sum(self.embeddings ** 2, axis=0)
                - 2 * similarity
        )

        # Derive the indices for minimum distances
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices

    def get_config(self):
        return {
            'num_embeddings': self.num_embeddings,
            'embedding_dim': self.embedding_dim,
            'beta': self.beta
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Decoder(Model):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 name=None):
        super(Decoder, self).__init__(name=name)

        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self._dec1 = Layer.Conv2D(self._num_hiddens, kernel_size=(3, 3), strides=(1, 1), padding='same', name='dec_l1')

        self._residual_stack = ResidualStack(
            self._num_hiddens,
            self._num_residual_layers,
            self._num_residual_hiddens,
            name='resblock2'
        )

        self._dec2 = Layer.Conv2DTranspose(self._num_hiddens // 2, kernel_size=(4, 4), strides=(2, 2), padding='same',
                                           name='dec_l2')
        self._dec3 = Layer.Conv2DTranspose(3, kernel_size=(4, 4), strides=(2, 2), padding='same', name='dec_l3')

    def call(self, inputs, training=None, mask=None):
        h = self._dec1(inputs)
        h = self._residual_stack(h)
        h = tf.nn.relu(self._dec2(h))
        reconstruction = self._dec3(h)
        return reconstruction


def get_vqvae(shape, latent_dim, num_embeddings, model_encoder, model_decoder):
    inputs = Input(shape=shape)
    vq_layer = VectorQuantizer(latent_dim, num_embeddings, name="vector_quantizer")
    encoder_outputs = model_encoder(inputs)
    quantized_latents = vq_layer(encoder_outputs)
    reconstructions = model_decoder(quantized_latents)
    return Model(inputs, reconstructions, name="vq_vae")


class VQVAETrainer(Model):
    def __init__(self, input_shape, num_hiddens, num_residual_layers, num_residual_hiddens, train_variance, latent_dim,
                 num_embeddings, name=None):
        super(VQVAETrainer, self).__init__(name=name)
        self.train_variance = train_variance
        self.latent_dim = latent_dim  # embedding_dim
        self.num_embeddings = num_embeddings
        self.num_hiddens = num_hiddens

        self.model_encoder = Encoder(num_hiddens=self.num_hiddens, num_residual_layers=num_residual_layers,
                                     num_residual_hiddens=num_residual_hiddens, latent_dim=self.latent_dim,
                                     name='Encoder')
        self.model_decoder = Decoder(num_hiddens=self.num_hiddens, num_residual_layers=num_residual_layers,
                                     num_residual_hiddens=num_residual_hiddens, name='Decoder')

        self.vqvae = get_vqvae(
            shape=input_shape,
            latent_dim=self.latent_dim,
            num_embeddings=self.num_embeddings,
            model_encoder=self.model_encoder,
            model_decoder=self.model_decoder)

        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = Mean(name="vq_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]

    def train_step(self, x):
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions = self.vqvae(x)

            # Calculate the losses.
            reconstruction_loss = (tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance)
            total_loss = reconstruction_loss + sum(self.vqvae.losses)

        # Backpropagation
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # Loss tracking
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        # Log results.
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
