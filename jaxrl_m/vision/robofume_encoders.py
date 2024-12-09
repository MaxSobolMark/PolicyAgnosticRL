import jax.numpy as jnp
from flax import linen as nn


class RobofumeEncoder(nn.Module):
    @nn.compact
    def __call__(self, x, train: bool):
        dtype = jnp.float32
        x = x.astype(dtype) / 255.0 - 0.5
        layers = [
            # Layer 1
            nn.Conv(32, (3, 3), (2, 2), padding="VALID"),
            nn.LayerNorm(epsilon=1e-5, dtype=dtype),
            nn.relu,
            # Layer 2
            nn.Conv(32, (3, 3), (1, 1), padding="VALID"),
            nn.LayerNorm(epsilon=1e-5, dtype=dtype),
            nn.relu,
            # Layer 3
            nn.Conv(32, (3, 3), (1, 1), padding="VALID"),
            nn.LayerNorm(epsilon=1e-5, dtype=dtype),
            nn.relu,
            # Layer 4
            nn.Conv(32, (3, 3), (1, 1), padding="VALID"),
            nn.LayerNorm(epsilon=1e-5, dtype=dtype),
            nn.relu,
        ]

        for layer in layers:
            x = layer(x)

        # x = x.squeeze()
        # x = x.reshape(x.shape[:-3] + (-1,))
        # return x
        if len(x.shape) == 3:
            return x.reshape((-1,))
        # elif len(x.shape) == 6:
        #     # ddpm with shape (batch_size, repeat, history, height, width, channels)
        #     return x.reshape((x.shape[0], x.shape[1], -1))
        return x.reshape((x.shape[0], -1))


robofume_configs = {"robofume": RobofumeEncoder}
