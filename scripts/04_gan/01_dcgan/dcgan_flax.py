import argparse
import shutil
import time
from functools import partial
from pathlib import Path
from typing import Tuple, Any, Dict

import os

from notebooks.utils import display

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import optax
import tensorflow as tf
from flax.training import train_state
from jax import jit
from jax._src.basearray import ArrayLike, Array
from tqdm import trange

from keras import utils
import jax
import jax.numpy as jnp
from flax import linen as nn

from scripts.common import rng_seq, conv_layers, deconv_layers
import tensorflow_datasets as tfds


class TrainState(train_state.TrainState):
  batch_stats: Any


class Generator(nn.Module):
  channels: int

  @nn.compact
  def __call__(self, latent_vectors: jnp.ndarray, training=False):
    latent_vectors = jnp.reshape(latent_vectors, newshape=(-1, 1, 1, 100))

    def post_conv(inputs: ArrayLike) -> ArrayLike:
      inputs = nn.BatchNorm(use_running_average=True, momentum=0.9)(inputs)
      return nn.leaky_relu(inputs, 0.2)

    return deconv_layers(latent_vectors,
                         filters=(512, 256, 128, 64, self.channels),
                         kernel_sizes=(4,) * 5,
                         strides=(1, 2, 2, 2, 2),
                         paddings=("VALID", "SAME", "SAME", "SAME", "SAME"),
                         use_biases=(False,) * 5,
                         post_op_cbs=(post_conv,
                                      post_conv,
                                      post_conv,
                                      post_conv,
                                      nn.tanh),
                         )


class Discriminator(nn.Module):

  @nn.compact
  def __call__(self, images: ArrayLike, training: bool = False):
    """
    output is a sigmoid score in [0, 1]
    """

    def batch_norm(layer: nn.Module):
      return nn.BatchNorm(use_running_average=True, momentum=0.9)(layer)

    def activation_dropout(input_layer: ArrayLike):
      input_layer = nn.leaky_relu(input_layer, 0.2)
      return nn.Dropout(rate=0.3)(input_layer, deterministic=not training)

    def all(input_layer):
      return activation_dropout(batch_norm(input_layer))

    conv_output = conv_layers(images,
                              filters=(64, 128, 256, 512, 1),
                              kernel_sizes=(4,) * 5,
                              strides=(2, 2, 2, 2, 1),
                              paddings=("SAME", "SAME", "SAME", "SAME", "VALID"),
                              use_biases=(False,) * 5,
                              post_op_cbs=(
                                activation_dropout,
                                all,
                                all,
                                all,
                                nn.sigmoid
                              ),
                              )
    flattened = jnp.reshape(conv_output, newshape=(-1,))

    return flattened

@partial(jit, static_argnames=["batch_size", "latent_dim"])
def generator_train_step(generator_state: TrainState, discriminator_state: TrainState, batch_size: int, latent_dim: int,
                         rng_key: Array):
  rng_gen = rng_seq(key=rng_key)
  latent = jax.random.normal(next(rng_gen), shape=(batch_size, latent_dim))

  def loss_fn(generator_params: Dict[str, Any]) -> Tuple[Array, Dict[str, Array]]:
    generated_images, mutables = generator_state.apply_fn(
      {"params": generator_params, "batch_stats": generator_state.batch_stats}, latent, mutable=["batch_stats"],
    )

    # in this step we only care about the mutables from the generator
    sigmoid_score, _ = discriminator_state.apply_fn(
      {"params": discriminator_state.params, "batch_stats": discriminator_state.batch_stats}, images=generated_images,
      training=False,
      mutable=["batch_stats"])

    sigmoid_score = jax.lax.stop_gradient(sigmoid_score)

    # I think that, because the target is always 1 here, we don't have to worry about log(0) here
    # like we do in other cases. However, in the text they add noise to the labels. I'm not sure how necessary that is.
    """
    Text: A useful trick when training GANs is to add a small amount of random noise to the training labels. This helps 
    to improve the stability of the training process and sharpen the generated images. This label smoothing acts as way 
    to tame the discriminator, so that it is presented with a more challenging task and doesn’t overpower the generator.

    Oops, actually this is for the discriminator    
    
    """
    loss = -jnp.mean(jnp.log(sigmoid_score))

    return loss, mutables

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, mutables), grads = grad_fn(generator_state.params)

  new_generator_state = generator_state.apply_gradients(grads=grads, batch_stats=mutables["batch_stats"])

  return new_generator_state, loss


@partial(jit, static_argnames=["latent_dim"])
def discriminator_train_step(generator_state: TrainState, discriminator_state: TrainState, real_images: Array,
                             latent_dim: int, rng_key: Array):
  batch_size = real_images.shape[0]
  rng_gen = rng_seq(key=rng_key)
  latent = jax.random.normal(next(rng_gen), shape=(batch_size, latent_dim))
  generated_images, _ = generator_state.apply_fn(
    {"params": generator_state.params, "batch_stats": generator_state.batch_stats}, latent, mutable=["batch_stats"])

  images = jnp.concatenate([real_images, generated_images])
  # there are multiple approachs to label smoothing
  real_labels = jnp.ones(shape=(batch_size,)) - 0.1 * jax.random.uniform(next(rng_gen), shape=(batch_size,))
  fake_labels = jnp.zeros_like(real_labels) + 0.1 * jax.random.uniform(next(rng_gen), shape=real_labels.shape)
  labels = jnp.concatenate([real_labels, fake_labels])

  def loss_fn(discriminator_params: Dict[str, Array], images, labels, rng_key: Array) -> \
    Tuple[Array, Dict[str, Array]]:
    rng_gen = rng_seq(key=rng_key)

    sigmoid_score, mutables = discriminator_state.apply_fn(
      {"params": discriminator_params, "batch_stats": discriminator_state.batch_stats}, images,
      training=True,
      rngs={"dropout": next(rng_gen)},
      mutable=['batch_stats'])

    sigmoid_score = jnp.clip(sigmoid_score, 1e-7, 1.0 - 1e-7)
    ce = -(labels * jnp.log(sigmoid_score) + (1 - labels) * jnp.log(1 - sigmoid_score))
    loss = jnp.mean(ce)
    return loss, mutables

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, mutables), grads = grad_fn(discriminator_state.params, images, labels, next(rng_gen))
  new_discriminator_state = discriminator_state.apply_gradients(grads=grads, batch_stats=mutables["batch_stats"])

  return new_discriminator_state, loss


def train_step(generator_state: TrainState, discriminator_state: TrainState, batch: Array, batch_size: int,
               latent_dim: int, rng_key: Array) -> Tuple[TrainState, TrainState, Dict[str, float]]:
  rng_gen = rng_seq(key=rng_key)
  generator_state, generator_loss = generator_train_step(generator_state=generator_state,
                                                         discriminator_state=discriminator_state,
                                                         batch_size=batch_size, latent_dim=latent_dim,
                                                         rng_key=next(rng_gen))
  discriminator_state, discriminator_loss = discriminator_train_step(generator_state=generator_state,
                                                                     discriminator_state=discriminator_state,
                                                                     real_images=batch, latent_dim=latent_dim,
                                                                     rng_key=next(rng_gen))

  return generator_state, discriminator_state, {"generator_loss": float(generator_loss),
                                                "discriminator_loss": float(discriminator_loss)}


def preprocess(img: tf.Tensor) -> jnp.ndarray:
  # inputs are in [0, 256]
  # different preprocessing step than with mnist. This one is
  # centered on 0 and range from [-1, 1].
  # Ah this is so that we can use tanh on the output rather than sigmoid
  return (tf.cast(img, tf.float32) - 127.5) / 127.5


def postprocess(img: Array) -> Array:
  return img * 127.5 + 127.5


@jit
def eval_step(generator_state: TrainState, discriminator_state: TrainState, latent: Array, real_images: Array) -> Tuple[
  Dict[str, float], Array]:
  generated_images, _ = generator_state.apply_fn(
    {"params": generator_state.params, "batch_stats": generator_state.batch_stats}, latent, training=False,
    mutable=["batch_stats"])
  fake_scores, _ = discriminator_state.apply_fn(
    {"params": discriminator_state.params, "batch_stats": discriminator_state.batch_stats},
    generated_images, training=False, mutable=["batch_stats"])
  real_scores, _ = discriminator_state.apply_fn(
    {"params": discriminator_state.params, "batch_stats": discriminator_state.batch_stats},
    real_images, training=False, mutable=["batch_stats"])

  return {"fake_scores": float(jnp.mean(fake_scores)), "real_scores": float(jnp.mean(real_scores))}, postprocess(
    generated_images)


if __name__ == "__main__":

  """
  TODO:
  - [x] Double-check the loss. In the book example they're using tanh as the output, but I think my loss was for sigmoid.
  Also, shouldn't there be a negative on the cross-entropy?
  - [x] Check the normalization of the input images
  - [x] Output metrics
  - [x] Add an eval step
  - [ ] Load and save models
  - [x] Handle the dropout rng as per https://flax.readthedocs.io/en/latest/guides/flax_sharp_bits.html#flax-linen-dropout-layer-and-randomness
  - [x] dataset loading isn't working properly.
  - [x] Discriminator loss value is not changing.
  - [ ] Discriminator label smoothing is not working properly, causes problems w/ loss function.
  - [ ] After a few steps we get nan. Generator seems good. Discriminator step seems to be the problem.
  """

  parser = argparse.ArgumentParser()
  parser.add_argument("mode", choices=["train", "reconstruct", "generate"])
  parser.add_argument("--seed", type=int, default=None)
  args = parser.parse_args()

  seed = args.seed if args.seed is not None else time.time_ns()

  rng_gen = rng_seq(seed=seed)

  IMAGE_SIZE = 64
  CHANNELS = 1
  BATCH_SIZE = 128
  Z_DIM = 100
  EPOCHS = 300
  LOAD_MODEL = False
  ADAM_BETA_1 = 0.5
  ADAM_BETA_2 = 0.999
  LEARNING_RATE = 0.0002
  NOISE_PARAM = 0.1

  train_ds: tf.data.Dataset = utils.image_dataset_from_directory(
    "./data/lego-brick-images/dataset",
    labels=None,
    color_mode="grayscale",
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
    interpolation="bilinear",
  )

  steps_per_epoch = len(train_ds)
  train_ds = train_ds.repeat()
  train_ds = train_ds.map(lambda x: preprocess(x))
  train_ds = iter(tfds.as_numpy(train_ds))

  generator = Generator(channels=CHANNELS)
  print(generator.tabulate(jax.random.key(0), jnp.ones((2, Z_DIM)),
                           console_kwargs={"width": 120}))  # , compute_flops=True, compute_vjp_flops=True))
  output, variables = generator.init_with_output(next(rng_gen), jnp.empty((2, Z_DIM)))
  assert output.shape == (2, IMAGE_SIZE, IMAGE_SIZE, CHANNELS), "Generator output shape fail"

  generator_state = TrainState.create(
    apply_fn=generator.apply,
    params=variables["params"],
    tx=optax.adam(learning_rate=LEARNING_RATE, b1=ADAM_BETA_1, b2=ADAM_BETA_2),
    batch_stats=variables["batch_stats"]
  )

  discriminator = Discriminator()
  print(discriminator.tabulate(jax.random.key(0), jnp.ones((2, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)), training=False,
                               console_kwargs={"width": 200}))
  output, variables = discriminator.init_with_output(next(rng_gen), jnp.empty((2, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)))
  assert output.shape == (2,), "Discriminator output shape fail"

  discriminator_state = TrainState.create(
    apply_fn=discriminator.apply,
    params=variables["params"],
    tx=optax.adam(learning_rate=LEARNING_RATE, b1=ADAM_BETA_1, b2=ADAM_BETA_2),
    batch_stats=variables["batch_stats"]
  )

  output_dir = Path("data/dcgan_flax").absolute()
  if output_dir.exists():
    shutil.rmtree(output_dir)
  output_dir.mkdir(parents=True, exist_ok=False)

  eval_latent = jax.random.normal(next(rng_gen), shape=(BATCH_SIZE, Z_DIM))
  eval_images = next(train_ds)

  match args.mode:
    case "train":

      for epoch in range(EPOCHS):
        # or rate_noinv_fmt
        for _ in (pbar := trange(steps_per_epoch,
                                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_inv_fmt}]")):
          batch = next(train_ds)
          generator_state, discriminator_state, metrics = train_step(generator_state, discriminator_state, batch,
                                                                     batch_size=BATCH_SIZE, latent_dim=Z_DIM,
                                                                     rng_key=next(rng_gen))

          pbar.set_description(f"{metrics}")

        print("eval_step")
        metrics, generated_images = eval_step(generator_state, discriminator_state, eval_latent, eval_images)
        print(f"eval_metrics: {metrics}")
        display(generated_images, save_to=output_dir / f"eval_{epoch + 1}")

        # metrics, reconstruction = eval_step(state.params, test_ds, next(rng_gen))
        # checkpoint_manager.save(epoch, {"state": state}, metrics={k: float(v) for k, v in metrics.items()})
        # display(reconstruction)

    # case "reconstruct":
    #     example_images = x_test[:5000]
    #     reconstruction = predict(state.params, example_images, next(rng_gen))
    #     print('inputs')
    #     display(example_images)
    #     print('predictions')
    #     display(reconstruction)
