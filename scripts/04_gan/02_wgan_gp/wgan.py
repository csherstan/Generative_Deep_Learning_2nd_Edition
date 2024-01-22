import argparse
import shutil
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Tuple, Dict, Any

import os

import numpy as np

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".50"
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import jax
import optax
from flax import struct
from flax.metrics import tensorboard
from flax.core.scope import VariableDict
from flax.struct import dataclass
from flax.training.train_state import TrainState
from jax import Array, jit
from tqdm import trange

from notebooks.utils import display
from scripts.common import rng_seq, preprocess_image_tanh, conv_layers, deconv_layers

from keras import utils
import jax.numpy as jnp
import tensorflow_datasets as tfds

import flax.linen as nn


class Generator(nn.Module):
    channels: int
    latent_dim: int

    @nn.compact
    def __call__(self, latent_vectors: jnp.ndarray, training: bool):
        x = jnp.reshape(latent_vectors, newshape=(-1, 1, 1, self.latent_dim))

        x = nn.ConvTranspose(features=512, kernel_size=(4, 4), strides=(1, 1), padding="VALID", use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not training, epsilon=0.001, momentum=0.9)(x)
        x = nn.leaky_relu(x, 0.2)

        x = nn.ConvTranspose(features=256, kernel_size=(4, 4), strides=(2, 2), padding="SAME", use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not training, epsilon=0.001, momentum=0.9)(x)
        x = nn.leaky_relu(x, 0.2)

        x = nn.ConvTranspose(features=128, kernel_size=(4, 4), strides=(2, 2), padding="SAME", use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not training, epsilon=0.001, momentum=0.9)(x)
        x = nn.leaky_relu(x, 0.2)

        x = nn.ConvTranspose(features=64, kernel_size=(4, 4), strides=(2, 2), padding="SAME", use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not training, epsilon=0.001, momentum=0.9)(x)
        x = nn.leaky_relu(x, 0.2)

        x = nn.ConvTranspose(features=self.channels, kernel_size=(4, 4), strides=(2, 2), padding="SAME", use_bias=True)(x)
        x = nn.tanh(x)

        return x

class GeneratorState(TrainState):
    batch_stats: VariableDict


def create_generator_state(latent_size: int, channels: int, learning_rate: float, rng_key: Array) -> GeneratorState:
    rng_gen = rng_seq(key=rng_key)
    generator = Generator(channels, latent_size)
    # print(generator.tabulate(next(rng_gen), jnp.ones((2, latent_size)), console_kwargs={"soft_wrap": True, "width": 120}))
    output, variables = generator.init_with_output(next(rng_gen), jnp.zeros((2, latent_size)), training=False)
    return GeneratorState.create(
        apply_fn=generator.apply,
        params=variables["params"],
        tx=optax.adam(learning_rate=learning_rate, b1=0.5, b2=0.9),
        batch_stats=variables["batch_stats"],
    )


class Critic(nn.Module):

    @nn.compact
    def __call__(self, images: Array, training: bool, rng_key: Array):
        rng_gen = rng_seq(key=rng_key)

        def activation_dropout(input_layer: Array):
            input_layer = nn.leaky_relu(input_layer, 0.2)
            return nn.Dropout(rate=0.3)(input_layer, deterministic=not training, rng=next(rng_gen))

        conv_output = conv_layers(images,
                                  filters=(64, 128, 256, 512, 1),
                                  kernel_sizes=(4,) * 5,
                                  strides=(2, 2, 2, 2, 1),
                                  paddings=("SAME", "SAME", "SAME", "SAME", "VALID"),
                                  use_biases=(True,) * 5,
                                  post_op_cbs=(
                                      activation_dropout,
                                      activation_dropout,
                                      activation_dropout,
                                      activation_dropout,
                                      None
                                  ),
                                  )
        flattened = jnp.reshape(conv_output, newshape=(-1,))

        return flattened


def create_critic_state(image_shape: Tuple[int, int, int], learning_rate: float, rng_key: Array) -> TrainState:
    rng_gen = rng_seq(key=rng_key)
    critic = Critic()
    # print(critic.tabulate(next(rng_gen), jnp.ones((2, *image_shape)), console_kwargs={"soft_wrap": True, "width": 120}))
    output, variables = critic.init_with_output(next(rng_gen), jnp.ones((2, *image_shape)), training=False,
                                                rng_key=next(rng_gen))
    return TrainState.create(
        apply_fn=critic.apply,
        params=variables["params"],
        tx=optax.adam(learning_rate=learning_rate, b1=0.5, b2=0.9)
    )


@dataclass
class WCGANState():
    generator_state: GeneratorState
    critic_state: TrainState


# @partial(jit, static_argnames=["batch_size", "latent_dim"])
def generator_train_step(wcgan_state: WCGANState, batch_size: int,
                         latent_dim: int,
                         rng_key: Array):
    rng_gen = rng_seq(key=rng_key)
    latent = jax.random.normal(next(rng_gen), shape=(batch_size, latent_dim))
    generator_state = wcgan_state.generator_state
    critic_state = wcgan_state.critic_state

    def loss_fn(generator_params: VariableDict) -> Tuple[Array, Dict[str, Array]]:
        generated_images, mutables = generator_state.apply_fn(
            {"params": generator_params, "batch_stats": generator_state.batch_stats}, latent, training=True,
            mutable=["batch_stats"],
        )

        # in this step we only care about the mutables from the generator
        score = critic_state.apply_fn(
            {"params": critic_state.params},
            images=generated_images,
            training=True,
            rng_key=next(rng_gen),
        )

        loss = -jnp.mean(score)
        return loss, mutables

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, mutables), grads = grad_fn(generator_state.params)

    generator_state = generator_state.apply_gradients(grads=grads, batch_stats=mutables["batch_stats"])

    return WCGANState(generator_state=generator_state, critic_state=critic_state), loss


# @partial(jit, static_argnames=["latent_dim", "gp_weight"])
def critic_step(wcgan_state: WCGANState, real_images: Array, latent_dim: int,
                gp_weight: float, rng_key: Array) -> Tuple[
    WCGANState, Dict[str, Array]]:
    rng_gen = rng_seq(key=rng_key)
    batch_size = real_images.shape[0]
    latents = jax.random.normal(next(rng_gen), shape=(batch_size, latent_dim))
    generator_state = wcgan_state.generator_state
    critic_state = wcgan_state.critic_state

    generated_images, generator_mutables = generator_state.apply_fn(
        {"params": generator_state.params,
         "batch_stats": generator_state.batch_stats},
        latents,
        training=True,
        mutable=["batch_stats"])
    generator_state = generator_state.replace(batch_stats=generator_mutables["batch_stats"])

    def critic_loss(critic_params: VariableDict, real_images: Array, generated_images: Array, gp_weight: float,
                    rng_key: Array) -> Tuple[
        Array, Dict[str, Array]]:
        rng_gen = rng_seq(key=rng_key)
        real_scores = critic_state.apply_fn({"params": critic_params}, real_images, training=True,
                                            rng_key=next(rng_gen))
        generated_scores = critic_state.apply_fn({"params": critic_params}, generated_images, training=True,
                                                 rng_key=next(rng_gen))

        """
        Wasserstein loss: -\frac{1}{n}\sum_{i=1}^{n}(y_{i}p_{i}
        where y_i is the label in {-1, 1} and p_i is the score for element i
        """
        metrics = {}
        real_loss = -real_scores.mean()
        metrics["c_real_loss"] = real_loss
        generated_loss = generated_scores.mean()
        metrics["c_generated_loss"] = generated_loss
        ws_loss = (real_loss + generated_loss)  # + ((real_scores + generated_scores).sum()/(2*batch_size))**2
        metrics["c_wass_loss"] = ws_loss

        # create a set of interpolated images.
        # this caught me the first time... I was taking the rnd over the whole shape, but I just want a single
        # value for each number so that I can move along the diff vector.
        # For a 3D image this is [batch, 1, 1, 1]
        alpha = jax.random.uniform(next(rng_gen),
                                  shape=(generated_images.shape[0],) + (1,) * len(generated_images.shape[1:]))
        diff = generated_images - real_images
        interpolated = real_images + alpha * diff

        def gradient_penalty_fn(interpolated_images: Array, critic_params: VariableDict) -> Array:
            """
            Interpolated images that are a small random distance along the vector between real_image and generated_image
            pairs.

            Take the grad of the score and penalizing for differing from 1.
            """
            return critic_state.apply_fn({"params": critic_params}, interpolated_images, training=True,
                                         rng_key=next(rng_gen)).mean()

        gradient_penalty_grad = jax.grad(gradient_penalty_fn)(interpolated, critic_params)
        norm = jnp.sum(jnp.square(gradient_penalty_grad), axis=[1, 2, 3])
        gradient_penalty = ((1 - norm) ** 2).mean()

        metrics["c_gp"] = gradient_penalty

        loss = ws_loss + gp_weight * gradient_penalty
        metrics["c_loss"] = loss

        return loss, metrics

    # slow warning comes from here
    (loss, metrics), grads = jax.value_and_grad(critic_loss, has_aux=True)(critic_state.params, real_images=real_images,
                                                                           generated_images=generated_images,
                                                                           gp_weight=gp_weight, rng_key=next(rng_gen))
    critic_state = critic_state.apply_gradients(grads=grads)

    return WCGANState(generator_state=generator_state, critic_state=critic_state), metrics


def train_step(wcgan_state: WCGANState, real_images: Array, latent_dim: int, critic_iterations: int, gp_weight: float,
               rng_key: Array) -> \
    Tuple[WCGANState, Dict[str, Array]]:
    rng_gen = rng_seq(key=rng_key)
    batch_size = real_images.shape[0]

    metrics = {}
    critic_metrics = {}

    for i in range(critic_iterations):
        wcgan_state, new_metrics = critic_step(wcgan_state=wcgan_state,
                                               real_images=real_images,
                                               latent_dim=latent_dim,
                                               gp_weight=gp_weight,
                                               rng_key=next(rng_gen))
        for k, v in new_metrics.items():
            critic_metrics.setdefault(k, []).append(v)

    for k, v in critic_metrics.items():
        metrics[k] = jnp.array(v).mean()

    wcgan_state, loss = generator_train_step(wcgan_state=wcgan_state,
                                             batch_size=batch_size, latent_dim=latent_dim,
                                             rng_key=next(rng_gen))

    metrics["g_loss"] = loss

    return wcgan_state, metrics


def postprocess(img: Array) -> Array:
    return img * 127.5 + 127.5


@jit
def eval_step(generator_state: GeneratorState, discriminator_state: TrainState, latent: Array, real_images: Array) -> \
    Tuple[
        Dict[str, Array], Array]:
    rng_gen = rng_seq(seed=0)
    generated_images = generator_state.apply_fn(
        {"params": generator_state.params, "batch_stats": generator_state.batch_stats}, latent, training=False)
    fake_scores = discriminator_state.apply_fn({"params": discriminator_state.params}, generated_images,
                                               training=False, rng_key=next(rng_gen))
    real_scores = discriminator_state.apply_fn({"params": discriminator_state.params}, real_images,
                                               training=False, rng_key=next(rng_gen))

    return {"fake_scores": jnp.mean(fake_scores), "real_scores": jnp.mean(real_scores)}, postprocess(
        generated_images)


def sample_batch(ds: tf.data.Dataset) -> np.ndarray:
    batch = ds.take(1).get_single_element()
    return batch.numpy()
def main():
    IMAGE_SIZE = 64
    CHANNELS = 3
    BATCH_SIZE = 512
    Z_DIM = 128
    EPOCHS = 200
    LOAD_MODEL = False
    LEARNING_RATE = 0.0002
    CRITIC_ITERATIONS = 3
    GP_WEIGHT = 10.0
    STEPS_PER_EPOCH = 2
    ADAM_BETA_1 = 0.5
    ADAM_BETA_2 = 0.9

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "reconstruct", "generate"])
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else time.time_ns()

    rng_gen = rng_seq(seed=seed)

    # train_ds: tf.data.Dataset = utils.image_dataset_from_directory(
    #     "./data/lego-brick-images/dataset",
    #     labels=None,
    #     color_mode="grayscale",
    #     image_size=(IMAGE_SIZE, IMAGE_SIZE),
    #     batch_size=BATCH_SIZE,
    #     shuffle=True,
    #     seed=42,
    #     interpolation="bilinear",
    # )

    train_ds: tf.data.Dataset = utils.image_dataset_from_directory(
        "data/celeba-dataset/img_align_celeba/img_align_celeba",
        labels=None,
        color_mode="rgb",
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42,
        interpolation="bilinear",
    )

    steps_per_epoch = STEPS_PER_EPOCH  # len(train_ds)
    train_ds = train_ds.repeat()
    train_ds = train_ds.map(lambda x: preprocess_image_tanh(x))

    output_dir = Path(f"output/wgan_flax/{datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')}").absolute()
    output_dir.mkdir(parents=True, exist_ok=False)

    summary_writer = tensorboard.SummaryWriter(output_dir / "tensorboard")

    model_state = WCGANState(
        generator_state=create_generator_state(Z_DIM, CHANNELS, LEARNING_RATE, next(rng_gen)),
        critic_state=create_critic_state((IMAGE_SIZE, IMAGE_SIZE, CHANNELS), LEARNING_RATE, next(rng_gen)),
    )

    # metrics = train_step(model_state, real_images=jnp.stack([jnp.ones(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))*0.2, jnp.ones(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))*-0.2]),
    #            latent_dim=Z_DIM,
    #            critic_iterations=CRITIC_ITERATIONS,
    #            gp_weight=GP_WEIGHT,
    #            rng_key=next(rng_gen))
    # print(metrics)

    eval_latent = jax.random.normal(next(rng_gen), shape=(BATCH_SIZE, Z_DIM))
    eval_images = sample_batch(train_ds)
    display(eval_images, save_to=output_dir / f"eval_baseline")

    match args.mode:
        case "train":

            for epoch in range(EPOCHS):
                metrics = {}
                for _ in range(steps_per_epoch):
                    batch = jnp.array(sample_batch(train_ds))
                    model_state, step_metrics = train_step(model_state, real_images=batch, latent_dim=Z_DIM,
                                                           critic_iterations=CRITIC_ITERATIONS,
                                                           gp_weight=GP_WEIGHT,
                                                           rng_key=next(rng_gen))

                    for k, v in step_metrics.items():
                        metrics.setdefault(k, []).append(v)
                    # pbar.set_description(f"{metrics}")

                for k, v in metrics.items():
                    summary_writer.scalar(tag=f"epoch_{k}", value=jnp.array(v).mean().item(), step=epoch)
                summary_writer.flush()

                print("eval_step")
                metrics, generated_images = eval_step(generator_state=model_state.generator_state,
                                                      discriminator_state=model_state.critic_state, latent=eval_latent,
                                                      real_images=eval_images)
                print(f"eval_metrics: {metrics}")
                for k, v in metrics.items():
                    summary_writer.scalar(tag=f"eval.{k}", value=v.item(), step=epoch)
                display(generated_images, save_to=output_dir / f"eval_{epoch + 1}")

        # case "reconstruct":
        #     example_images = x_test[:5000]
        #     reconstruction = predict(state.params, example_images, next(rng_gen))
        #     print('inputs')
        #     display(example_images)
        #     print('predictions')
        #     display(reconstruction)


if __name__ == '__main__':
    main()
