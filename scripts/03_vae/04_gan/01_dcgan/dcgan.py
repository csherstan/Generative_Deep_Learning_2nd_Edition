import os
from typing import Tuple, cast

os.environ["KERAS_BACKEND"] = "jax"

from scripts.common import conv_layers, deconv_layers

import keras_core as K
from keras_core import layers, models, datasets, losses, metrics, utils, Layer, optimizers
from keras_core.src import backend
import jax
import jax.dlpack
import jax.numpy as jnp

import multiprocessing.pool
import tensorflow as tf


def tensor_to_jax(tensor: tf.Tensor) -> jax.Array:
  """
  # TODO: this should probably be moved to ctrlax if it doesn't already exist.
  convert a tf tensor to a jax tensor
  @param tensor:
  @return:
  """
  return cast(jax.Array, jax.dlpack.from_dlpack(tf.experimental.dlpack.to_dlpack(tensor)))


def preprocess(img: tf.Tensor) -> jnp.ndarray:
  # different preprocessing step than with mnist. This one is
  # centered on 0 and range from [-1, 1].
  # Ah this is so that we can use tanh on the output rather than sigmoid
  return (tf.cast(img, tf.float32) - 127.5) / 127.5


"""
Question: when is better to make this a custom model vs. just using the base class?
Maybe use the custom model when it's the class that will be having `fit` called on it.
"""


def make_discriminator(input_shape: Tuple[int, ...]) -> models.Model:
  discriminator_input = layers.Input(shape=input_shape)

  def batch_norm(layer):
    return layers.BatchNormalization(momentum=0.9)(layer)

  def activation_dropout(input_layer: Layer):
    input_layer = layers.LeakyReLU(0.2)(input_layer)
    return layers.Dropout(0.3)(input_layer)

  def all(input_layer):
    return activation_dropout(batch_norm(input_layer))

  discriminator_output = conv_layers(layer_input=discriminator_input,
                                     filters=(64, 128, 256, 512, 1),
                                     kernel_sizes=(4, 4, 4, 4, 4),
                                     strides=(2, 2, 2, 2, 1),
                                     paddings=("same", "same", "same", "same", "valid"),
                                     activations=(None, None, None, None, "sigmoid"),
                                     use_biases=(False,) * 5,
                                     post_op_cbs=(
                                       activation_dropout,
                                       all,
                                       all,
                                       all,
                                       None
                                     ),
                                     )

  discriminator_output = layers.Flatten()(discriminator_output)

  return models.Model(discriminator_input, discriminator_output)


def make_generator(z_dim: int, channels: int) -> models.Model:
  generator_input = layers.Input(shape=(z_dim,))
  generator_output = layers.Reshape((1, 1, z_dim))(generator_input)

  def post_conv(layer: layers.Layer) -> layers.Layer:
    layer = layers.BatchNormalization(momentum=0.9)(layer)
    return layers.LeakyReLU(0.2)(layer)

  generator_output = deconv_layers(
    layer_input=generator_output,
    filters=(512, 256, 128, 64, channels),
    kernel_sizes=(4, 4, 4, 4, 4),
    strides=(1, 2, 2, 2, 2),
    activations=(None, None, None, None, "tanh"),
    use_biases=(False,) * 5,
    paddings=("valid", "same", "same", "same", "same"),
    post_op_cbs=(
      post_conv,
      post_conv,
      post_conv,
      post_conv,
      None
    )
  )

  return models.Model(generator_input, generator_output)


# class DCGAN(models.Model):
#
#     def __init__(self, generator, **kwargs):
#         super().__init__(**kwargs)
#         self.generator = generator
#         self.seed_generator = K.random.SeedGenerator(1337)
#
#     def call(self, inputs, training=False):
#         K.random.normal(shape=[100, 2], seed=self.seed_generator)

class SamplingLayer(Layer):
  def __init__(self, size: int, **kwargs):
    super().__init__(**kwargs)
    self.size = size
    self.seed_gen = K.random.SeedGenerator(1337)

  def call(self, count):
    return K.random.normal(shape=(count, self.size), seed=self.seed_gen)


def extract_vars_from_parent(trainable_variables, non_trainable_variables, parent: Layer, child: Layer):
  # TODO: the mapping should be static, so, instead, you could just return indices and apply them as needed.
  child_trainable_variables_map = {}
  for idx, v in enumerate(child.trainable_variables):
    child_trainable_variables_map[id(v)] = idx

  o_trainable_variables = [0] * len(child.trainable_variables)
  found = 0
  for var, val in zip(parent.trainable_variables, trainable_variables):
    if id(var) in child_trainable_variables_map:
      o_trainable_variables[child_trainable_variables_map[id(var)]] = val
      found += 1

    if found == len(o_trainable_variables):
      break

  child_non_trainable_variables_map = {}
  for idx, v in enumerate(child.non_trainable_variables):
    child_non_trainable_variables_map[id(v)] = idx

  o_non_trainable_variables = [0] * len(child.non_trainable_variables)
  found = 0
  for var, val in zip(parent.non_trainable_variables, non_trainable_variables):
    if id(var) in child_non_trainable_variables_map:
      o_non_trainable_variables[child_non_trainable_variables_map[id(var)]] = val
      found += 1
    if found == len(o_non_trainable_variables):
      break

  return o_trainable_variables, o_non_trainable_variables


class DCGAN(models.Model):

  def __init__(self, discriminator: models.Model, generator: models.Model, latent_dim: int, noise_param: float,
               **kwargs):
    super(DCGAN, self).__init__(**kwargs)
    self.sampling_layer = SamplingLayer(latent_dim, name="sampling_layer")
    self.disciminator = discriminator
    self.generator = generator
    self.latent_dim = latent_dim
    self.noise_param = noise_param

    self.d_optimizer = optimizers.Optimizer(learning_rate=LEARNING_RATE)
    self.g_optimizer = optimizers.Optimizer(learning_rate=LEARNING_RATE)

  # so with the stateless model we can only have a single Optimizer past in so if we want to do
  # complicated things with multiple optimizers we need to create a custom optimizer class.
  # def compile(
  #     self,
  #     optimizer,
  #     **kwargs
  # ):
  #     super(DCGAN, self).compile(**kwargs)
  #     self.
  #     self.d_optimizer = d_optimizer
  #     self.g_optimizer = g_optimizer
  #     self.d_loss_metric = metrics.Mean(name="d_loss")
  #     self.d_real_acc_metric = metrics.BinaryAccuracy(name="d_real_acc")
  #     self.d_fake_acc_metric = metrics.BinaryAccuracy(name="d_fake_acc")
  #     self.d_acc_metric = metrics.BinaryAccuracy(name="d_acc")
  #     self.g_loss_metric = metrics.Mean(name="g_loss")
  #     self.g_acc_metric = metrics.BinaryAccuracy(name="g_acc")

  # @property
  # def metrics(self):
  #     return [
  #         self.d_loss_metric,
  #         self.d_real_acc_metric,
  #         self.d_fake_acc_metric,
  #         self.d_acc_metric,
  #         self.g_loss_metric,
  #         self.g_acc_metric,
  #     ]

  def train_step(self, state, real_images: jnp.ndarray):
    # These are just the values of the variables, not the variables themselves
    (
      trainable_variables,
      non_trainable_variables,
      optimizer_variables,
      metrics_variables,
    ) = state

    """
    Because jax.value_and_grad only optimizes wrt a single loss, I haven't figured out a super clean way to do this. 
    3 options I can think of.
    1. Just run the network twice, once for each loss
    2. Compute per-sample gradients (there's an example of doing this on the Jax page), then partition out the gradients
    to apply them appropriately.
    3. First compute the loss for the generator, return the generated images and then compute the discriminator loss 
    """

    def compute_discriminator_loss(trainable_variables, non_trainable_variables, real_images, generated_images):
      real_predictions = self.disciminator.stateless_call(trainable_variables, non_trainable_variables, real_images,
                                                          training=True)
      fake_predictions = self.disciminator.stateless_call(trainable_variables, non_trainable_variables,
                                                          generated_images, training=True)

      real_labels = K.ops.ones_like(real_predictions)
      real_noisy_labels = real_labels + self.noise_param * K.random.uniform(K.ops.shape(real_predictions))

      fake_labels = K.ops.zeros_like(fake_predictions)
      fake_noisy_labels = fake_labels + self.noise_param * K.random.uniform(K.ops.shape(fake_predictions))

      real_discriminator_loss = jnp.mean(losses.binary_crossentropy(real_noisy_labels, real_predictions))
      fake_discriminator_loss = jnp.mean(losses.binary_crossentropy(fake_noisy_labels, fake_predictions))

      total_discriminator_loss = (real_discriminator_loss + fake_discriminator_loss) / 2

      return total_discriminator_loss, fake_predictions

    def compute_generator_loss(trainable_variables, non_trainable_variables, count: int):
      result, non_trainable_variables = self.stateless_call(trainable_variables, non_trainable_variables, count)

      predictions = result["scoring"]

      real_labels = K.ops.ones(shape=(count, ))
      real_noisy_labels = real_labels + self.noise_param * K.random.uniform(K.ops.shape(predictions),
                                                                            seed=self.generator)

      generator_loss = jnp.mean(losses.binary_crossentropy(real_noisy_labels, predictions))

      return generator_loss, result

    # def compute_full_loss(trainable_variables, non_trainable_variables, real_images, random_latent_vectors):
    #     generated_images = self.generator(random_latent_vectors, training=True)
    #     d_gfun = jax.value_and_grad(compute_discriminator_loss)
    #     (discriminator_loss, fake_predictions), discriminator_grads = d_gfun(trainable_variables, non_trainable_variables, real_images, generated_images)
    #
    #     real_labels = K.ops.ones_like(fake_predictions)
    #     real_noisy_labels = real_labels + self.noise_param*K.random.uniform(K.ops.shape(fake_predictions), seed=self.generator)
    #
    #     generator_loss = jnp.mean(losses.binary_crossentropy(real_noisy_labels, fake_predictions))
    #
    #     return generator_loss, discriminator_grads

    # with backend.StatelessScope(
    #     state_mapping=mapping, collect_losses=return_losses
    # ) as scope:

    #
    # sampling_layer_trainable_vars, sampling_layer_non_trainable_variables = extract_vars_from_parent(
    #   trainable_variables, non_trainable_variables, self, self.sampling_layer)
    #
    # # TODO: we would now need to push the vars back into the state for when we return them
    # # right now I see that the random var is always the same, so the state of the SeedGenerator is not
    # # being updated. I think I need to: 1) check that the VAE code is actually doing the right thing and 2) create
    # # a simplified test example first.
    # random_latent_vectors, sampling_layer_state = self.sampling_layer.stateless_call(sampling_layer_trainable_vars, sampling_layer_non_trainable_variables,
    #                                    real_images)

    batch_size = K.ops.convert_to_tensor(K.ops.shape(real_images)[0])
    g_grad_fn = jax.value_and_grad(compute_generator_loss, has_aux=True)
    (g_loss, g_aux), g_grads = g_grad_fn(trainable_variables, non_trainable_variables, batch_size)

    d_grad_fn = jax.value_and_grad(compute_discriminator_loss, has_aux=True)
    (d_loss, d_aux), d_grads = d_grad_fn(trainable_variables, non_trainable_variables, real_images=real_images, generated_images=g_aux["generated_images"])





    #     (trainable_variables, optimizer_variables) = self.d_optimizer.stateless_apply(
    #         optimizer_variables, discriminator_grads, trainable_variables
    #     )
    #
    #     (trainable_variables, optimizer_variables) = self.g_optimizer.stateless_apply(
    #         optimizer_variables, generator_grads, trainable_variables
    #     )
    #
    #     # non_trainable_variables = aux_results["non_trainable_variables"]
    #     #
    #     # (
    #     #     trainable_variables,
    #     #     optimizer_variables,
    #     # ) = self.optimizer.stateless_apply(
    #     #     optimizer_variables, grads, trainable_variables
    #     # )
    #     #
    #     # # Update metrics.
    #     new_metrics_vars = []
    logs = {}
    #     # for metric in self.metrics:
    #     #     this_metric_vars = metrics_variables[
    #     #                        len(new_metrics_vars): len(new_metrics_vars) + len(metric.variables)
    #     #                        ]
    #     #     this_metric_vars = metric.stateless_update_state(this_metric_vars, aux_results["metrics"][metric.name])
    #     #     logs[metric.name] = metric.stateless_result(this_metric_vars)
    #     #     new_metrics_vars += this_metric_vars
    #
    #     # Return metric logs and updated state variables.
    #     state = (
    #         trainable_variables,
    #         non_trainable_variables,
    #         optimizer_variables,
    #         new_metrics_vars,
    #     )
    return logs, state

  def call(self, inputs, training=False):
    # TODO: a call doesn't need actual input data.
    random_latent_vectors = self.sampling_layer(inputs)
    generated_images = self.generator(random_latent_vectors, training=training)
    scoring = self.disciminator(generated_images, training=training)
    return {"random_latent_vectors": random_latent_vectors, "generated_images": generated_images, "scoring": scoring}


if __name__ == "__main__":
  IMAGE_SIZE = 64
  CHANNELS = 1
  BATCH_SIZE = 2
  Z_DIM = 1
  EPOCHS = 300
  LOAD_MODEL = False
  ADAM_BETA_1 = 0.5
  ADAM_BETA_2 = 0.999
  LEARNING_RATE = 0.0002
  NOISE_PARAM = 0.1

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
  # TODO: figure out the right way to handle this.
  # train_data = jnp.array([preprocess(t) for t in train_data])

  generator = make_generator(z_dim=Z_DIM, channels=CHANNELS)
  discriminator = make_discriminator((IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
  # dcgan = DCGAN(generator)
  # dcgan.compile(optimizer="adam")
  dcgan = DCGAN(discriminator=discriminator, generator=generator, latent_dim=Z_DIM, noise_param=NOISE_PARAM)

  # dcgan.compile(d_optimizer=optimizers.Optimizer(learning_rate=LEARNING_RATE),
  #               g_optimizer=optimizers.Optimizer(learning_rate=LEARNING_RATE))
  dcgan.compile(optimizer="adam")
  dcgan.run_eagerly = True

  # seed_generator = K.random.SeedGenerator(1337)
  # random_latent_vectors = K.random.normal(shape=(100, Z_DIM), seed=seed_generator)
  # generated = generator(random_latent_vectors)
  # predictions = discriminator(generated)

  with jax.disable_jit():
    dcgan.fit(
      train_data,
      epochs=EPOCHS,
    )
