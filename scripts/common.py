from typing import Sequence, Tuple, Generator, Callable, Optional, Union

import jax
import tensorflow as tf
from flax import linen as nn
from jax import numpy as jnp, Array
from jax._src.basearray import ArrayLike


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


def conv_layers(
  x: Array,
  filters: Sequence[int],
  kernel_sizes: Sequence[Union[Tuple[int, int], int]],
  strides: Sequence[int],
  paddings: Sequence[str],
  use_biases: Optional[Sequence[bool]] = None,
  post_op_cbs: Optional[Sequence[Callable[[ArrayLike], ArrayLike]]] = None,
) -> Array:
  if use_biases is None:
    use_biases = (True,) * len(filters)

  if post_op_cbs is None:
    post_op_cbs = (None,) * len(filters)

  for (filter, kernel_size, stride, padding, use_bias, post_op_cb) in zip(
    filters, kernel_sizes, strides, paddings, use_biases, post_op_cbs
  ):
    x = nn.Conv(
      features=filter,
      kernel_size=(kernel_size, kernel_size),
      strides=(stride, stride),
      padding=padding,
      use_bias=use_bias,
    )(x)

    if post_op_cb:
      x = post_op_cb(x)

  return x


def preprocess_image_tanh(img: tf.Tensor) -> jnp.ndarray:
  # inputs are in [0, 256]
  # different preprocessing step than with mnist. This one is
  # centered on 0 and range from [-1, 1].
  # Ah this is so that we can use tanh on the output rather than sigmoid
  return (tf.cast(img, tf.float32) - 127.5) / 127.5


def deconv_layers(
  x: Array,
  filters: Sequence[int],
  kernel_sizes: Sequence[Union[Tuple[int, int], int]],
  strides: Sequence[int],
  paddings: Sequence[str],
  use_biases: Optional[Sequence[bool]] = None,
  post_op_cbs: Optional[Sequence[Callable[[ArrayLike], ArrayLike]]] = None,
) -> Array:
  use_biases = use_biases or (True,) * len(filters)
  post_op_cbs = post_op_cbs or (None,) * len(filters)

  for (filter, kernel_size, stride, padding, use_bias, post_op_cb) in zip(filters, kernel_sizes, strides,
                                                                                      paddings, use_biases,
                                                                                      post_op_cbs):
    x = nn.ConvTranspose(features=filter, kernel_size=(kernel_size, kernel_size), strides=(stride, stride),
                         padding=padding,
                         use_bias=use_bias)(x)
    if post_op_cb:
      x = post_op_cb(x)

  return x
