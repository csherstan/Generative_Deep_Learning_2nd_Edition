import os

os.environ["KERAS_BACKEND"] = "jax"

import argparse
import time
from typing import Tuple

from notebooks.utils import display
from scripts.common import conv_layers, preprocess_mnist, make_deconv_decoder, rng_seq

import keras_core as K
from keras_core import layers, models, datasets, losses, metrics
import jax
import jax.numpy as jnp

"""
Note to self: I started trying to make the returns from the network Dataclasses, but it turns out Jax won't handle
them. There are probably ways to handle this (flax appears to offer a solution and I found a package that jax-dataclasses
or something like that) but for now I'm not going to worry about it. Note that NamedTuple should work.
"""


class SamplingLayer(layers.Layer):

    def __init__(self):
        super().__init__()
        self.seed_generator = K.random.SeedGenerator(1337)

    def call(self, inputs: Tuple[jax.Array, jax.Array]):
        z_mean, z_log_var = inputs
        # it's import to use K.ops.shape here rather than calling shape on the tensor directly.
        # Otherwise you will get errors like: TypeError: Failed to convert object of type <class 'something'> to Tensor
        batch = K.ops.shape(z_mean)[0]
        dim = K.ops.shape(z_mean)[1]
        return z_mean + K.ops.exp(0.5 * z_log_var) * K.random.normal(shape=(batch, dim), seed=self.seed_generator)

    def compute_output_shape(self, input_shape):
        return input_shape[0]


def make_encoder(output_size: int):
    encoder_input = layers.Input(shape=(32, 32, 1), name="encoder_input")
    x = conv_layers(layer_input=encoder_input,
                    filters=(32, 64, 128),
                    kernel_sizes=((3, 3),) * 3,
                    strides=(2,) * 3,
                    activations=("relu",) * 3,
                    paddings=("same",) * 3,
                    )
    conv_output_shape = x.shape[1:]
    x = layers.Flatten()(x)
    z_mean = layers.Dense(2, name="z_mean")(x)
    z_log_var = layers.Dense(2, name="z_log_var")(x)
    z = SamplingLayer()([z_mean, z_log_var])
    return models.Model(encoder_input, {"mean": z_mean, "log_var": z_log_var, "z": z},
                        name="encoder"), conv_output_shape


class VAE(models.Model):
    def __init__(self, encoder, decoder, beta: float, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta

        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker
        ]

    def call(self, inputs, training=False):
        encoder_output = self.encoder(inputs)
        reconstruction = self.decoder(encoder_output["z"])
        return {"mean": encoder_output["mean"], "log_var": encoder_output["log_var"], "reconstruction": reconstruction}

    def compute_reconstruction_loss(self, original, reconstruction):
        return self.beta*jnp.mean(losses.binary_crossentropy(original, reconstruction, axis=(1, 2, 3)))

    def compute_kl_loss(self, z_mean, z_log_var):
        return jnp.mean(jnp.sum(-0.5 * (1 + z_log_var - jnp.square(z_mean) - jnp.exp(z_log_var)), axis=1))

    def compute_loss_and_updates(self, trainable_variables, non_trainable_variables, inputs, training=False):
        result, non_trainable_variables = self.stateless_call(trainable_variables, non_trainable_variables, inputs,
                                                              training=training)
        reconstruction, z_mean, z_log_var = result["reconstruction"], result["mean"], result["log_var"]
        reconstruction_loss = self.compute_reconstruction_loss(inputs, reconstruction)
        kl_loss = self.compute_kl_loss(z_mean, z_log_var)
        total_loss = reconstruction_loss + kl_loss
        return total_loss, {"reconstruction": reconstruction,
                            "z_mean": z_mean,
                            "z_log_var": z_log_var,
                            "non_trainable_variables": non_trainable_variables,
                            "metrics": {
                                "reconstruction_loss": reconstruction_loss,
                                "kl_loss": kl_loss,
                                "total_loss": total_loss,
                            }
                            }

    def train_step(self, state, inputs):

        data = inputs[0]

        # This method comes from TensorFlowTrainer (inherited in Model)
        # https://keras.io/keras_core/guides/custom_train_step_in_jax/
        (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables,
        ) = state

        grad_fn = jax.value_and_grad(self.compute_loss_and_updates, has_aux=True)
        (loss, aux_results), grads = grad_fn(
            trainable_variables,
            non_trainable_variables,
            data,
            training=True,
        )

        non_trainable_variables = aux_results["non_trainable_variables"]

        (
            trainable_variables,
            optimizer_variables,
        ) = self.optimizer.stateless_apply(
            optimizer_variables, grads, trainable_variables
        )

        # Update metrics.
        new_metrics_vars = []
        logs = {}
        for metric in self.metrics:
            this_metric_vars = metrics_variables[
                               len(new_metrics_vars): len(new_metrics_vars) + len(metric.variables)
                               ]
            this_metric_vars = metric.stateless_update_state(this_metric_vars, aux_results["metrics"][metric.name])
            logs[metric.name] = metric.stateless_result(this_metric_vars)
            new_metrics_vars += this_metric_vars

        # Return metric logs and updated state variables.
        state = (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            new_metrics_vars,
        )
        return logs, state

    def test_step(self, state, data):
        x, y = data
        (
            trainable_variables,
            non_trainable_variables,
            metrics_variables,
        ) = state

        loss, aux_results = self.compute_loss_and_updates(trainable_variables=trainable_variables,
                                                          non_trainable_variables=non_trainable_variables, inputs=x,
                                                          training=False)

        # Update metrics.
        new_metrics_vars = []
        logs = {}
        for metric in self.metrics:
            this_metric_vars = metrics_variables[
                               len(new_metrics_vars): len(new_metrics_vars) + len(metric.variables)
                               ]
            this_metric_vars = metric.stateless_update_state(this_metric_vars, aux_results["metrics"][metric.name])
            logs[metric.name] = metric.stateless_result(this_metric_vars)
            new_metrics_vars += this_metric_vars

        state = (
            trainable_variables,
            non_trainable_variables,
            new_metrics_vars,
        )

        return logs, state


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "reconstruct", "generate"])
    parser.add_argument("--model")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--embedding_dim", type=int, default=2)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--beta", type=int, default=500)
    args = parser.parse_args()
    # TODO: this should also be use with the models
    seed = args.seed if args.seed is not None else time.time_ns()
    rng_gen = rng_seq(seed)

    IMAGE_SIZE = args.image_size
    BATCH_SIZE = args.batch_size
    EMBEDDING_SIZE = args.embedding_dim
    EPOCH = args.epoch
    BETA = args.beta

    callbacks = []
    checkpoint_path = "./checkpoint/model.{epoch:02d}.keras"
    checkpointer = K.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=False,
        save_freq="epoch",
        monitor="total_loss",
        mode="min",
        save_best_only=True,
        verbose=0,
    )
    callbacks.append(checkpointer)

    callbacks.append(K.callbacks.TensorBoard(log_dir="./logs"))

    # np.ndarray
    (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
    x_train = preprocess_mnist(x_train)
    x_test = preprocess_mnist(x_test)

    encoder_model, conv_shape = make_encoder(EMBEDDING_SIZE)
    decoder = make_deconv_decoder(encoding_size=EMBEDDING_SIZE, encoder_conv_output_shape=conv_shape)

    autoencoder = VAE(encoder_model, decoder, beta=BETA)

    if args.model is not None:
        autoencoder.load_weights(args.model)

    match args.mode:
        case "train":
            autoencoder.compile(optimizer="adam")
            autoencoder.fit(
                x_train,
                x_train,
                epochs=EPOCH,
                batch_size=BATCH_SIZE,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=callbacks,
            )
        case "reconstruct":
            example_images = x_test[:5000]
            forward_result = autoencoder.predict(example_images)
            print('inputs')
            display(example_images)
            print('predictions')
            display(forward_result["reconstruction"])

        case "generate":
            example_images = x_test[:5000]
            embeddings = encoder_model.predict(example_images)
            mins, maxs = jnp.min(embeddings, axis=0), jnp.max(embeddings, axis=0)
            sample = jax.random.uniform(next(rng_gen), shape=(18, EMBEDDING_SIZE), minval=mins, maxval=maxs)
            predictions = decoder.predict(sample)
            display(predictions)
