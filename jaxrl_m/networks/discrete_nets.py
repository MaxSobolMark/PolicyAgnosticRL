import distrax
import flax.linen as nn
import jax.numpy as jnp

from jaxrl_m.common.typing import *
from jaxrl_m.networks.mlp import MLP


class DiscreteQ(nn.Module):
    encoder: nn.Module
    qfunc: nn.Module

    def __call__(self, observations):
        latents = self.encoder(observations)
        return self.qfunc(latents)


class DiscreteCriticHead(nn.Module):
    hidden_dims: Sequence[int]
    n_actions: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.swish

    def setup(self):
        self.q = MLP((*self.hidden_dims, self.n_actions), activations=self.activation)

    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        return self.q(observations)


class DiscretePolicy(nn.Module):
    hidden_dims: Sequence[int]
    n_actions: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.swish

    @nn.compact
    def __call__(self, observations: jnp.ndarray, temperature=1.0) -> jnp.ndarray:
        logits = MLP((*self.hidden_dims, self.n_actions), activation=self.activation)(
            observations
        )
        return distrax.Categorical(logits=logits / temperature)
