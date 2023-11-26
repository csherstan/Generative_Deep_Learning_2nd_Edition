import argparse
import time

import os

from notebooks.utils import display
from scripts.common import conv_layers, preprocess_mnist, rng_seq, make_deconv_decoder

os.environ["KERAS_BACKEND"] = "jax"
import keras_core as K
from keras_core import layers, models, datasets
import jax
import jax.numpy as jnp


def make_encoder(output_size: int):
    encoder_input = layers.Input(shape=(32, 32, 1), name="encoder_input")
    x = conv_layers(
        encoder_input,
        filters=(32, 64, 128),
        kernel_sizes=((3, 3), (3, 3), (3, 3)),
        strides=(2, 2, 2),
        activations=("relu", "relu", "relu"),
        paddings=("same", "same", "same"),
    )
    conv_shape = x.shape[1:]
    x = layers.Flatten()(x)
    encoder_output = layers.Dense(output_size, name="encoder_output")(x)
    encoder = models.Model(encoder_input, encoder_output)

    return encoder, encoder_output, conv_shape


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "reconstruct", "generate"])
    parser.add_argument("--model")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    # TODO: this should also be use with the models
    seed = args.seed if args.seed is not None else time.time_ns()
    rng_gen = rng_seq(seed)

    callbacks = []
    checkpoint_path = "./checkpoint/model.{epoch:02d}.keras"
    checkpointer = K.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=False,
        save_freq="epoch",
        monitor="loss",
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

    encoding_size = 2
    encoder_model, encoder_output, conv_shape = make_encoder(encoding_size)
    decoder = make_deconv_decoder(encoding_size=2, encoder_conv_output_shape=conv_shape)

    autoencoder = models.Model(encoder_model.inputs, decoder(encoder_output))

    if args.model is not None:
        autoencoder.load_weights(args.model)

    match args.mode:
        case "train":
            autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
            autoencoder.fit(
                x_train,
                x_train,
                epochs=5,
                batch_size=100,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=callbacks,
            )
        case "reconstruct":
            example_images = x_test[:5000]
            predictions = autoencoder.predict(example_images)
            print('inputs')
            display(example_images)
            print('predictions')
            display(predictions)


        case "generate":
            example_images = x_test[:5000]
            embeddings= encoder_model.predict(example_images)
            mins, maxs = jnp.min(embeddings, axis=0), jnp.max(embeddings, axis=0)
            sample = jax.random.uniform(next(rng_gen), shape=(18, encoding_size), minval=mins, maxval=maxs)
            predictions = decoder.predict(sample)
            display(predictions)



