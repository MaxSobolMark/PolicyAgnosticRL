from typing import Any, Sequence

import flax.linen as nn
import jax.numpy as jnp
import jax.random as random
from jaxlib.xla_extension import DeviceArray

from jaxrl_m.common.common import MLP

ModuleDef = Any


class Encoder(nn.Module):
    hidden_dims: Sequence[int]
    latent_dim: int

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        h = MLP(self.hidden_dims)(observations)
        mean = MLP([self.latent_dim])(h)
        logvar = MLP([self.latent_dim])(h)
        return mean, logvar


class Decoder(nn.Module):
    hidden_dims: Sequence[int]
    output_dim: int

    @nn.compact
    def __call__(self, latent: jnp.ndarray) -> jnp.ndarray:
        output = MLP((*self.hidden_dims, self.output_dim))(latent)
        return output


class CVAE(nn.Module):
    encoder_hidden_dims: Sequence[int]
    latent_dim: int
    decoder_hidden_dims: Sequence[int]
    output_dim: int

    def setup(self):
        self.encoder = Encoder(self.encoder_hidden_dims, self.latent_dim)
        self.decoder = Decoder(self.decoder_hidden_dims, self.output_dim)

    def __call__(
        self, observations: jnp.ndarray, goals: jnp.ndarray, seed: DeviceArray
    ):
        rets = dict()

        combined = jnp.concatenate([observations, goals], axis=-1)
        rets["mean"], rets["logvar"] = self.encoder(combined)
        stds = jnp.exp(0.5 * rets["logvar"])
        z = rets["mean"] + stds * random.normal(seed, rets["mean"].shape)
        rets["reconstruction"] = self.decoder(z)

        return rets
