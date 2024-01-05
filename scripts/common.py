from typing import Sequence, Tuple, Generator, Callable, List, Optional, Union

import jax
from jax import numpy as jnp, Array
from keras_core import layers, models, Layer


def conv_layers(
    layer_input,
    filters: Sequence[int],
    kernel_sizes: Sequence[Union[Tuple[int, int], int]],
    strides: Sequence[int],
    activations: Sequence[str],
    paddings: Sequence[str],
    use_biases: Optional[Sequence[bool]] = None,
    post_op_cbs: Optional[Sequence[Callable[[Layer], Layer]]] = None,
):
    if use_biases is None:
        use_biases = (True, )*len(filters)

    if post_op_cbs is None:
        post_op_cbs = (None,)*len(filters)

    for (filter, kernel_size, stride, activation, padding, use_bias, post_op_cb) in zip(
        filters, kernel_sizes, strides, activations, paddings, use_biases, post_op_cbs
    ):
        layer_input = layers.Conv2D(
            filters=filter,
            kernel_size=kernel_size,
            strides=stride,
            activation=activation,
            padding=padding,
            use_bias=use_bias,
        )(layer_input)

        if post_op_cb:
            layer_input = post_op_cb(layer_input)

    return layer_input


def deconv_layers(
        layer_input,
        filters: Sequence[int],
        kernel_sizes: Sequence[Union[Tuple[int, int], int]],
        strides: Sequence[int],
        activations: Sequence[str],
        paddings: Sequence[str],
        use_biases: Optional[Sequence[bool]] = None,
        post_op_cbs: Optional[Sequence[Callable[[Layer], Layer]]] = None,
):
    use_biases = use_biases or (True, )*len(filters)
    post_op_cbs = post_op_cbs or (None, )*len(filters)

    layer = layer_input
    for (filter, kernel_size, stride, activation, padding, use_bias, post_op_cb) in zip(filters, kernel_sizes, strides, activations, paddings, use_biases, post_op_cbs):
        layer = layers.Conv2DTranspose(filters=filter, kernel_size=kernel_size, strides=stride, activation=activation, padding=padding)(layer)

        if post_op_cb:
            layer = post_op_cb(layer)
    return layer


def preprocess_mnist(imgs: jnp.ndarray) -> jnp.ndarray:
    imgs = imgs.astype(jnp.float32) / 255.0
    # first dim is batch size, so don't pad that
    imgs = jnp.pad(imgs, ((0, 0), (2, 2), (2, 2)), constant_values=0.0)
    imgs = jnp.expand_dims(imgs, -1)
    return imgs


def rng_seq(*, key: Array=None, seed: int=None) -> Generator[jax.Array, None, None]:
    if key is None:
        assert seed is not None
        key = jax.random.PRNGKey(seed)

    assert len(key) == 2

    while True:
        key, subkey = jax.random.split(key)
        yield subkey


def make_deconv_decoder(encoding_size: int, encoder_conv_output_shape: Tuple[int, ...]):
    decoder_input = layers.Input(shape=(encoding_size, ), name="decoder_input")
    x = layers.Dense(jnp.prod(jnp.array(encoder_conv_output_shape)).item())(decoder_input)
    x = layers.Reshape(encoder_conv_output_shape)(x)
    x = deconv_layers(x, (128, 64, 32), ((3, 3), (3, 3), (3, 3)), strides=(2, 2, 2), activations=("relu", "relu", "relu"), paddings=("same", "same", "same"))
    x = layers.Conv2D(1, (3,3), strides=1, activation="sigmoid", padding="same", name="decoder_output")(x)
    return models.Model(decoder_input, x)
