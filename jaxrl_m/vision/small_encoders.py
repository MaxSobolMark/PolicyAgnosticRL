"""From https://raw.githubusercontent.com/google/flax/main/examples/ppo/models.py"""

from multiprocessing import pool
from typing import Optional, Sequence, Tuple, Union

import jax.numpy as jnp
from flax import linen as nn


class AtariEncoder(nn.Module):
    """Class defining the actor-critic model."""

    @nn.compact
    def __call__(self, x):
        """Define the convolutional network architecture.

        Architecture originates from "Human-level control through deep reinforcement
        learning.", Nature 518, no. 7540 (2015): 529-533.
        Note that this is different than the one from  "Playing atari with deep
        reinforcement learning." arxiv.org/abs/1312.5602 (2013)

        Network is used to both estimate policy (logits) and expected state value;
        in other words, hidden layers' params are shared between policy and value
        networks, see e.g.:
        github.com/openai/baselines/blob/master/baselines/ppo1/cnn_policy.py
        """
        dtype = jnp.float32
        x = x.astype(dtype) / 255.0
        x = nn.Conv(
            features=32, kernel_size=(8, 8), strides=(4, 4), name="conv1", dtype=dtype
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            features=64, kernel_size=(4, 4), strides=(2, 2), name="conv2", dtype=dtype
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            features=64, kernel_size=(3, 3), strides=(1, 1), name="conv3", dtype=dtype
        )(x)
        x = nn.relu(x)
        x = x.reshape((*x.shape[:-3], -1))  # flatten
        return x


class SmallEncoder(nn.Module):
    features: Sequence[int] = (16, 16, 16)
    kernel_sizes: Sequence[int] = (3, 3, 3)
    strides: Sequence[int] = (1, 1, 1)
    padding: Union[Sequence[int], str] = (1, 1, 1)
    pool_method: Optional[str] = "max"
    pool_sizes: Sequence[int] = (2, 2, 1)
    pool_strides: Sequence[int] = (2, 2, 1)
    pool_padding: Sequence[int] = (0, 0, 0)
    bottleneck_dim: Optional[int] = None

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        assert len(self.features) == len(self.strides)

        x = observations.astype(jnp.float32) / 255.0

        for i in range(len(self.features)):

            if isinstance(self.padding, str):
                padding = self.padding
            else:
                padding = self.padding[i]

            x = nn.Conv(
                self.features[i],
                kernel_size=(self.kernel_sizes[i], self.kernel_sizes[i]),
                strides=(self.strides[i], self.strides[i]),
                padding=padding,
            )(x)
            if self.pool_method is not None:
                if self.pool_method == "avg":
                    pool_func = nn.avg_pool
                elif self.pool_method == "max":
                    pool_func = nn.max_pool
                x = pool_func(
                    x,
                    window_shape=(self.pool_sizes[i], self.pool_sizes[i]),
                    strides=(self.pool_strides[i], self.pool_strides[i]),
                    padding=((self.pool_padding[i], self.pool_padding[i]),) * 2,
                )
            x = nn.relu(x)

        if self.bottleneck_dim is not None:
            x = nn.Dense(self.bottleneck_dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)

        return x.reshape((*x.shape[:-3], -1))


small_configs = {"atari": AtariEncoder, "small": SmallEncoder}
