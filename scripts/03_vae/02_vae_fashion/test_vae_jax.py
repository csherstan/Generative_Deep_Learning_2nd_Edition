import unittest
import jax.numpy as jnp
from vae_jax import compute_reconstruction_loss


class MyTestCase(unittest.TestCase):
  def test_compute_reconstruction_loss(self):
    original = jnp.array([[1.0, 0.3, 0.1], [0.9, 0.0, 0.3]])
    reconstruction = jnp.array([[0.9, 0.2, 0.3], [1.0, 0.3, 0.0]])

    loss = compute_reconstruction_loss(original, reconstruction)

    # note that when I was calculating these numbers I originally used numpy, but jax and numpy
    # gave slightly different values for the same computation and it would fail the test.
    expected = -jnp.array([[-0.10536051565782628, -0.6390318596501768, -0.44140472997745284],
                          [-1.5942386, -0.35667494393873245, -4.835428765287499]])

    assert jnp.allclose(loss, expected)


if __name__ == '__main__':
  unittest.main()
