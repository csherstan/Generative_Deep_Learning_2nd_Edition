import os
import shutil
from pathlib import Path

import optax
from flax.training.train_state import TrainState
from jax import Array
from orbax.checkpoint import PyTreeCheckpointer, CheckpointManager, CheckpointManagerOptions
from tqdm import trange

os.environ["KERAS_BACKEND"] = "jax"

import argparse
import time
from typing import Tuple, Dict, Any, cast

from notebooks.utils import display
from scripts.common import preprocess_mnist, rng_seq

from keras_core import datasets
import jax
import jax.numpy as jnp

from flax import linen as nn

import tensorflow as tf
import tensorflow_datasets as tfds

"""
Note to self: I started trying to make the returns from the network Dataclasses, but it turns out Jax won't handle
them. There are probably ways to handle this (flax appears to offer a solution and I found a package that jax-dataclasses
or something like that) but for now I'm not going to worry about it. Note that NamedTuple should work.
"""

class SamplingLayer(nn.Module):

  def __call__(self, rng: Array, z_mean: Array, z_log_var: Array) -> Array:
    return z_mean + jnp.exp(0.5 * z_log_var) * jax.random.normal(key=rng, shape=z_mean.shape)


class Encoder(nn.Module):
  latent_size: int

  @nn.compact
  def __call__(self, inputs: Array) -> Tuple[Array, Array, Tuple[int, ...]]:
    x = inputs
    x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
    x = nn.relu(x)
    x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
    x = nn.relu(x)
    x = nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2), padding="SAME", name="final_conv")(x)
    x = nn.relu(x)
    conv_output_shape = x.shape
    x = jnp.reshape(x, (x.shape[0], -1))
    # x = jnp.ravel(x) # flatten
    z_mean = nn.Dense(self.latent_size, name="z_mean")(x)
    z_log_var = nn.Dense(self.latent_size, name="z_log_var")(x)

    return z_mean, z_log_var, conv_output_shape


# class Decoder(nn.Module):
#
#   conv_output_shape: Tuple[int, ...]
#
#   @nn.compact
#   # @partial(jax.jit, static_argnames=["conv_output_shape"])
#   def __call__(self, latent_vectors: Array) -> Array:
#     x = latent_vectors
#     x = nn.Dense(jnp.prod(jnp.array(self.conv_output_shape)))(x)
#     x = jnp.reshape(x, self.conv_output_shape)
#
#     x = nn.ConvTranspose(features=128, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
#     x = nn.relu(x)
#     x = nn.ConvTranspose(features=64, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
#     x = nn.relu(x)
#     x = nn.ConvTranspose(features=32, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
#     x = nn.relu(x)
#     x = nn.Conv(features=1, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
#     x = nn.sigmoid(x)
#
#     return x

class Decoder(nn.Module):

  conv_output_shape: Tuple[int, ...]

  def setup(self) -> None:
    size = 1
    for d in self.conv_output_shape:
      size *= d

    self.fc = nn.Dense(size)
    self.cnn = nn.Sequential([
      nn.ConvTranspose(features=128, kernel_size=(3, 3), strides=(2, 2), padding="SAME"),
      nn.relu,
      nn.ConvTranspose(features=64, kernel_size=(3, 3), strides=(2, 2), padding="SAME"),
      nn.relu,
      nn.ConvTranspose(features=32, kernel_size=(3, 3), strides=(2, 2), padding="SAME"),
      nn.relu,
      nn.Conv(features=1, kernel_size=(3, 3), strides=(1, 1), padding="SAME"),
      nn.sigmoid,
    ])
  def __call__(self, latent_vectors: Array) -> Array:
    x = latent_vectors
    x = self.fc(x)
    x = jnp.reshape(x, (-1, ) + self.conv_output_shape)
    return self.cnn(x)

class VAE(nn.Module):
  """
  In hindsight, I don't think that packaging everything together in this class was the right move - better
  to keep all the modules separate. That seems easier to deal with setting up the shapes dynamically, etc.
  """

  input_shape: Tuple[int, ...]
  latent_size: int

  def setup(self) -> None:
    self.encoder = Encoder(self.latent_size)
    # vars = self.encoder.init(jax.random.PRNGKey(0), jnp.empty((1,) + self.input_shape), capture_intermediates=True)

    # TODO: I really don't love this, but I'm not sure what the right approach is
    # final_conv_shape = vars["intermediates"]["final_conv"]["__call__"][0].shape[1:]
    self.decoder = Decoder((4, 4, 128))
    self.sampling_layer = SamplingLayer()

  def __call__(self, inputs: Array, z_rng: Array, training: bool =False) -> Tuple[Array, Array, Array]:
    z_mean, z_log_var, conv_output_shape = self.encoder(inputs)
    latent_vectors = self.sampling_layer(z_rng, z_mean, z_log_var)
    reconstruction = self.decoder(latent_vectors)  # , conv_output_shape)
    return z_mean, z_log_var, reconstruction


# @jax.jit
def compute_reconstruction_loss(original: Array, reconstruction: Array) -> Array:
  # in this implementation the sigmoid has already been applied.
  """
  CE = - E_p[log q] = -[p_1*log(q_1)+p_0*log(q_0)]
  let p_1 = original, p_0 = 1 - original
  let q_1 = reconstruction, q_0 = 1 - reconstruction

  We cannot use this form - in the case that q_1 is either 0 or 1, this would give -inf.
  I see that the keras implementation does exactly what I have, but clips the probability ouputs in the range
  [EPSILON, 1-EPSILON].

  Args:
      original:
      reconstruction:

  Returns:

  """
  epsilon = 1e-7
  reconstruction = jnp.clip(reconstruction, epsilon, 1.0 - epsilon)
  # for testing purposes and more flexibility in downstream use I think it's better not to take the mean here
  return -(original * jnp.log(reconstruction) + (1 - original) * jnp.log(1 - reconstruction))


# @jax.jit
def compute_kl_loss(z_mean: Array, z_log_var: Array) -> Array:
  return jnp.sum(-0.5 * (1 + z_log_var - jnp.square(z_mean) - jnp.exp(z_log_var)), axis=1) / z_mean.shape[1]


def loss_fn(params: Dict[str, Any], original: Array, beta: float, rng: Array) -> Tuple[Array, Dict[str, Dict[str, Array]]]:
  model = model_factory()
  z_mean, z_log_var, reconstruction = model.apply({'params': params}, original, rng)
  reconstruction_loss = beta * jnp.mean(compute_reconstruction_loss(original, reconstruction))
  kl_loss = jnp.mean(compute_kl_loss(z_mean, z_log_var))
  total_loss = reconstruction_loss + kl_loss

  return total_loss, {
    "metrics": {"reconstruction_loss": reconstruction_loss, "kl_loss": kl_loss, "total_loss": total_loss}}


# this jit made all the difference in training speed.
@jax.jit
def train_step(state: TrainState, batch: Array, rng: Array) -> Tuple[TrainState, Dict[str, Dict[str, Array]]]:
  (total_loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
    state.params,
    batch,
    500,  # TODO: how do I handle passing beta correctly?
    rng,
  )

  return state.apply_gradients(grads=grads), aux


@jax.jit
def eval_step(params: Dict[str, Any], batch: Array, z_rng: Array) -> Tuple[Dict[str, Array], Array]:
  model = model_factory()

  def eval_model(vae: VAE) -> Tuple[Dict[str, Array], Array]:
    z_mean, z_log_var, reconstruction = vae(batch, z_rng)
    reconstruction_loss = jnp.mean(compute_reconstruction_loss(batch, reconstruction))
    kl_loss = jnp.mean(compute_kl_loss(z_mean, z_log_var))
    return {"reconstruction_loss": reconstruction_loss, "kl_loss": kl_loss}, reconstruction

  return cast(Tuple[Dict[str, Array], Array], nn.apply(eval_model, model)({'params': params}))

def predict(params: Dict[str, Any], batch: Array, z_rng: Array) -> Array:
  model = model_factory()
  def _predict(vae: VAE) -> Array:
    z_mean, z_log_var, reconstruction = vae(batch, z_rng)
    return reconstruction

  return cast(Array, nn.apply(_predict, model)({'params': params}))


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
  seed = args.seed if args.seed is not None else time.time_ns()

  # TODO: I'm not sure about this approach, need to think it through more.
  rng_gen = rng_seq(seed=seed)

  IMAGE_SIZE = args.image_size
  BATCH_SIZE = args.batch_size
  EMBEDDING_SIZE = args.embedding_dim
  EPOCH = args.epoch
  BETA = args.beta
  CHANNEL = 1

  # np.ndarray
  (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
  x_train = preprocess_mnist(x_train)
  x_test = preprocess_mnist(x_test)

  train_set_size = x_train.shape[0]
  steps_per_epoch = int(x_train.shape[0] / BATCH_SIZE)

  train_ds = tf.data.Dataset.from_tensor_slices(x_train)
  train_ds = train_ds.shuffle(train_set_size, reshuffle_each_iteration=True)
  train_ds = train_ds.repeat()
  train_ds = train_ds.batch(BATCH_SIZE)
  train_ds = train_ds.prefetch(2)
  train_ds = iter(tfds.as_numpy(train_ds))

  test_ds = x_test[:100]

  IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, CHANNEL)

  autoencoder = VAE(IMAGE_SHAPE, EMBEDDING_SIZE)

  model_factory = lambda: VAE(IMAGE_SHAPE, EMBEDDING_SIZE)

  checkpoint_dir = Path("data/vae_jax").absolute()
  if checkpoint_dir.exists():
    shutil.rmtree(checkpoint_dir)

  checkpointer = PyTreeCheckpointer()
  checkpoint_manager = CheckpointManager(checkpoint_dir,
                                         checkpointer,
                                         options=CheckpointManagerOptions(create=True,
                                                                          max_to_keep=1,
                                                                          best_fn=lambda metrics: metrics["reconstruction_loss"],
                                                                          best_mode="min"))

  params = autoencoder.init(next(rng_gen), jnp.ones((1, ) + IMAGE_SHAPE), next(rng_gen))['params']
  # print(jax.tree_util.tree_map(lambda x: x.shape, params))
  state = TrainState.create(
    apply_fn=autoencoder.apply,
    params=params,
    tx=optax.adam(learning_rate=0.001)
  )

  if args.model:
    state = checkpointer.restore(args.model, item={"state": state})["state"]

  match args.mode:
    case "train":


      for epoch in range(EPOCH):
        # or rate_noinv_fmt
        for _ in (pbar := trange(steps_per_epoch,
                                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_inv_fmt}]")):
          batch = next(train_ds)
          state, aux = train_step(state, batch, next(rng_gen))
          pbar.set_description(f"{aux['metrics']}")

        print("eval_step")
        metrics, reconstruction = eval_step(state.params, test_ds, next(rng_gen))
        checkpoint_manager.save(epoch, {"state": state}, metrics={k: float(v) for k, v in metrics.items()})
        display(reconstruction)

    case "reconstruct":
        example_images = x_test[:5000]
        reconstruction = predict(state.params, example_images, next(rng_gen))
        print('inputs')
        display(example_images)
        print('predictions')
        display(reconstruction)
    #
    # case "generate":
    #     example_images = x_test[:5000]
    #     embeddings = encoder_model.predict(example_images)
    #     mins, maxs = jnp.min(embeddings, axis=0), jnp.max(embeddings, axis=0)
    #     sample = jax.random.uniform(next(rng_gen), shape=(18, EMBEDDING_SIZE), minval=mins, maxval=maxs)
    #     predictions = decoder.predict(sample)
    #     display(predictions)
