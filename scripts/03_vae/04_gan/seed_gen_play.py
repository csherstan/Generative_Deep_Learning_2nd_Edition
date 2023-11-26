import os
os.environ["KERAS_BACKEND"] = "jax"
from typing import Tuple, cast

import keras_core as K
from keras_core import layers, models, datasets, losses, metrics, utils, Layer, optimizers
from keras_core.src import backend
import jax
import jax.dlpack
import jax.numpy as jnp

import tensorflow as tf

class SamplingLayer(Layer):
  def __init__(self, size: int, **kwargs):
    super().__init__(**kwargs)
    self.size = size
    self.seed_gen = K.random.SeedGenerator(1337)

  def call(self, inputs):
    return K.random.normal(shape=(K.ops.shape(inputs)[0], self.size), seed=self.seed_gen)

class MyModel(models.Model):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.sampling_layer = SamplingLayer(2)


  def call(self, inputs, training=False):
    vecs = self.sampling_layer(inputs)
    return vecs

  def train_step(self, state, data):
    (
      trainable_variables,
      non_trainable_variables,
      optimizer_variables,
      metrics_variables,
    ) = state
    def compute_loss(trainable_variables, non_trainable_variables, inputs, training):
      vecs, non_trainable_variables = self.stateless_call(trainable_variables, non_trainable_variables, inputs, training=training)
      return (0., {"vecs": vecs, "non_trainable_variables": non_trainable_variables})

    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
    (loss, aux), grads = grad_fn(trainable_variables, non_trainable_variables, data, training=True)

    non_trainable_variables = aux["non_trainable_variables"]
    print(aux["vecs"])
    print(non_trainable_variables)

    state = (
      trainable_variables,
      non_trainable_variables,
      optimizer_variables,
      metrics_variables,
    )

    return {}, state

def preprocess(img: tf.Tensor) -> jnp.ndarray:
  # different preprocessing step than with mnist. This one is
  # centered on 0 and range from [-1, 1].
  # Ah this is so that we can use tanh on the output rather than sigmoid
  return (tf.cast(img, tf.float32) - 127.5) / 127.5


if __name__ == "__main__":

  # train_data = jnp.array([[0]*100])

  IMAGE_SIZE = 64
  BATCH_SIZE = 2
  train_data = utils.image_dataset_from_directory(
    "./data/lego-brick-images/dataset",
    labels=None,
    color_mode="grayscale",
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
    interpolation="bilinear",
  )

  train_data = train_data.map(lambda x: preprocess(x))

  model = MyModel()
  model.run_eagerly = True
  model.compile()

  with jax.disable_jit():
    model.fit(
      train_data,
      epochs = 1,
    )