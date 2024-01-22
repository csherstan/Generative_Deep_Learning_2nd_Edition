import os
from typing import Tuple, cast

os.environ["KERAS_BACKEND"] = "jax"

from scripts.common_keras import conv_layers, deconv_layers

import keras_core as K
from keras_core import layers, models, datasets, losses, metrics, utils, Layer, optimizers, StatelessScope
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


class LatentSamplingLayer(Layer):
  def __init__(self, size: int, **kwargs):
    super().__init__(**kwargs)
    self.size = size
    self.seed_gen = K.random.SeedGenerator(1337)

  def call(self, inputs):
    batch_size = K.ops.shape(inputs)[0]
    return K.random.normal(shape=(batch_size, self.size), seed=self.seed_gen)


def extract_var_mapping(parent: Layer, child: Layer):
  child_trainable_variables_map = {}
  for idx, v in enumerate(child.trainable_variables):
    child_trainable_variables_map[id(v)] = {"child_idx": idx}

  for idx, v, in enumerate(parent.trainable_variables):
    if id(v) in child_trainable_variables_map:
      child_trainable_variables_map[id(v)]["parent_idx"] = idx

  child_non_trainable_variables_map = {}
  for idx, v in enumerate(child.non_trainable_variables):
    child_non_trainable_variables_map[id(v)] = {"child_idx": idx}

  for idx, v in enumerate(parent.non_trainable_variables):
    if id(v) in child_non_trainable_variables_map:
      child_non_trainable_variables_map[id(v)]["parent_idx"] = idx

  return child_non_trainable_variables_map, child_non_trainable_variables_map

def extract_vars(child_trainable_variables_map, child_non_trainable_variables_map, trainable_variables, non_trainable_variables):
  trainable_variables_subset = [None]*len(child_trainable_variables_map)
  for k, v in child_trainable_variables_map.items():
    trainable_variables_subset[v["child_idx"]] = trainable_variables[v["parent_idx"]]

  non_trainable_variables_subset = [None]*len(child_non_trainable_variables_map)
  for k, v in child_non_trainable_variables_map.items():
    non_trainable_variables_subset[v["child_idx"]] = non_trainable_variables[v["parent_idx"]]

  return trainable_variables_subset, non_trainable_variables_subset

def insert_vars(child_trainable_variables_map, child_non_trainable_variables_map, trainable_variables, non_trainable_variables):

  for k, v in child_trainable_variables_map.items():
    trainable_variables[v["parent_idx"]] = child_trainable_variables_map[v["child_idx"]]

  for k, v in child_non_trainable_variables_map.items:
    non_trainable_variables[v["parent_idx"]] = child_non_trainable_variables_map[v["child_idx"]]

  return


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
    self.latent_sampling_layer = LatentSamplingLayer(latent_dim, name="sampling_layer")
    self.seed_gen = K.random.SeedGenerator(1337)
    self.noise_layer = layers.GaussianNoise(stddev=1)
    self.discriminator = discriminator
    self.generator = generator
    self.latent_dim = latent_dim
    self.noise_param = noise_param

    # TODO: what to do with the initial seed?


    self.d_grad_fn = jax.value_and_grad(self.compute_discriminator_loss, has_aux=True)
    self.g_grad_fn = jax.value_and_grad(self.compute_generator_loss, has_aux=True)

  # so with the stateless model we can only have a single Optimizer past in so if we want to do
  # complicated things with multiple optimizers we need to create a custom optimizer class.
  def compile(
      self,
      *args,
      **kwargs
  ):
      super(DCGAN, self).compile(*args, **kwargs)

      self.tvars_idx_map = {id(var): idx for idx, var in enumerate(self.trainable_variables)}

      self.d_loss_metric = metrics.Mean(name="d_loss")
      # self.d_real_acc_metric = metrics.BinaryAccuracy(name="d_real_acc")
      # self.d_fake_acc_metric = metrics.BinaryAccuracy(name="d_fake_acc")
      # self.d_acc_metric = metrics.BinaryAccuracy(name="d_acc")
      self.g_loss_metric = metrics.Mean(name="g_loss")
      # self.g_acc_metric = metrics.BinaryAccuracy(name="g_acc")

      self.mapped = {id(layer): extract_var_mapping(self, layer) for layer in [
        self.latent_sampling_layer,
        self.seed_gen,
        self.noise_layer,
        self.discriminator,
        self.generator,
      ]}

      pass

  @property
  def metrics(self):
      return [
          self.d_loss_metric,
          # self.d_real_acc_metric,
          # self.d_fake_acc_metric,
          # self.d_acc_metric,
          self.g_loss_metric,
          # self.g_acc_metric,
      ]

  def compute_discriminator_loss(self, trainable_variables, non_trainable_variables, real_images, generated_images):
    batch_size = K.ops.shape(real_images)[0]
    images = K.ops.concatenate([real_images, generated_images], axis=0)
    trainable_variables, non_trainable_variables = extract_vars_from_parent(trainable_variables, non_trainable_variables, self, self.discriminator)
    predictions, non_trainable_variables = self.discriminator.stateless_call(trainable_variables, non_trainable_variables, images, training=True)

    real_labels = K.ops.ones(shape=(batch_size, 1))
    real_noisy_labels = real_labels + self.noise_param * K.random.uniform(K.ops.shape(real_labels),
                                                                          seed=self.seed_gen)

    fake_labels = K.ops.zeros_like(real_labels)
    fake_noisy_labels = fake_labels + self.noise_param * K.random.uniform(K.ops.shape(fake_labels),
                                                                          seed=self.seed_gen)

    noisy_labels = K.ops.concatenate([real_noisy_labels, fake_noisy_labels], axis=0)

    discriminator_loss = jnp.mean(losses.binary_crossentropy(noisy_labels, predictions))

    return discriminator_loss, {"metrics": {"d_loss": discriminator_loss}}

  def compute_generator_loss(self, trainable_variables, non_trainable_variables, random_vectors):
    g_trainable_variables, g_non_trainable_variables = extract_vars_from_parent(trainable_variables, non_trainable_variables, self, self.generator)
    generated_images = self.generator.stateless_call(g_trainable_variables, g_non_trainable_variables, random_vectors)

    d_trainable_variables, d_non_trainable_variables = extract_vars_from_parent(trainable_variables, non_trainable_variables, self, self.discriminator)

    predictions = self.discriminator.stateless_call(d_trainable_variables, d_non_trainable_variables, generated_images)

    real_labels = K.ops.ones_like(predictions)
    real_noisy_labels = real_labels + self.noise_param * K.random.uniform(K.ops.shape(predictions),
                                                                          seed=self.seed_gen)

    generator_loss = jnp.mean(losses.binary_crossentropy(real_noisy_labels, predictions))

    return generator_loss, {"metrics": {"g_loss": generator_loss}}

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

    trainable_mapping = zip(self.trainable_variables, trainable_variables)
    non_trainable_mapping = zip(
      self.non_trainable_variables, non_trainable_variables
    )
    mapping = list(trainable_mapping) + list(non_trainable_mapping)

    batch_size = K.ops.convert_to_tensor(K.ops.shape(real_images)[0])

    # TODO: need a way to wrap this so that we: 1. first grab the correct vars and then 2. repack the non_trainable_variables
    random_vectors, non_trainable_variables = self.latent_sampling_layer.stateless_call(*extract_vars_from_parent(trainable_variables, non_trainable_variables, self, self.latent_sampling_layer), real_images)

    # with StatelessScope(state_mapping=mapping) as scope:
    #   """
    #   Pulling in the scope variable values is handled at the variable level
    #   ```
    #       def value(self):
    #         if in_stateless_scope():
    #             scope = get_stateless_scope()
    #             value = scope.get_current_value(self)
    #             if value is not None:
    #                 return self._maybe_autocast(value)
    #         if self._value is None:
    #             # Unitialized variable. Return a placeholder.
    #             # This is fine because it's only ever used
    #             # in during shape inference / graph tracing
    #             # (anything else would be a bug, to be fixed.)
    #             return self._maybe_autocast(
    #                 self._initializer(self._shape, dtype=self._dtype)
    #             )
    #         return self._maybe_autocast(self._value)
    #   ```
    #
    #   """
    #
    #   random_vectors = K.random.normal(shape=(batch_size, self.latent_dim), seed=self.seed_gen)
    generated_images, non_trainable_variables = self.generator.stateless_call(*extract_vars_from_parent(trainable_variables, non_trainable_variables, self, self.generator), random_vectors)

    # new_non_trainable_variables = []
    # for v in non_trainable_variables:
    #   new_v = scope.get_current_value(v)
    #   if new_v is not None:
    #     new_non_trainable_variables.append(new_v)
    #   else:
    #     new_non_trainable_variables.append(v)
    #
    # non_trainable_variables = new_non_trainable_variables

    metrics = {}

    (d_loss, aux), d_grads = self.d_grad_fn(trainable_variables, non_trainable_variables, real_images=real_images,
                                generated_images=generated_images)
    metrics.update(aux["metrics"])
    (g_loss, aux), g_grads = self.g_grad_fn(trainable_variables, non_trainable_variables, random_vectors)
    metrics.update(aux["metrics"])

    # I think that with the way that the stateless_api works we need to use a single optimizer.
    # I could either: 1. write a custom optimizer or 2. compile the grads from the discriminator and generator
    # 2. seems like the easiest at the moment, but I'm a bit annoyed that keras doesn't seem
    # to have really clean way to get the trainable variables by part.


    """
    Need to know which var from the original space maps to the current space
    """
    grads = [None]*len(g_grads)

    for v in self.generator.trainable_variables:
      idx = self.tvars_idx_map[id(v)]
      grads[idx] = g_grads[idx]

    for v in self.discriminator.trainable_variables:
      idx = self.tvars_idx_map[id(v)]
      grads[idx] = d_grads[idx]

    (trainable_variables, optimizer_variables) = self.optimizer.stateless_apply(
      optimizer_variables, grads, trainable_variables
    )

    # Update metrics.
    new_metrics_vars = []
    logs = {}
    for metric in self.metrics:
        this_metric_vars = metrics_variables[
                           len(new_metrics_vars): len(new_metrics_vars) + len(metric.variables)
                           ]
        this_metric_vars = metric.stateless_update_state(this_metric_vars, metrics[metric.name])
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

  def call(self, inputs, training=False):
    # I'm having a hard time figuring out the *right* way to do this.
    # It seems like I need to just make sure that all the network is accessed once in this call function
    # for when building happens so that weights are created. But in a multi-loss or multi-method model approach
    # this method doesn't seem overly useful actually.
    batch_size = K.ops.shape(inputs)[0]
    random_latent_vectors = self.latent_sampling_layer(inputs)
    generated_images = self.generator(random_latent_vectors, training=training)
    scoring = self.discriminator(generated_images, training=training)
    return {"random_latent_vectors": random_latent_vectors, "generated_images": generated_images, "scoring": scoring}


if __name__ == "__main__":
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

  # The JAX trainer assumes a single optimizer so we need to use a different approach.
  dcgan.compile(optimizer="adam")
  # dcgan.run_eagerly = True


  with jax.disable_jit():
    dcgan.fit(
      train_data,
      epochs=EPOCHS,
    )
