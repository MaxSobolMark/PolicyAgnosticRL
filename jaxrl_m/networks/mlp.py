from typing import Callable, Optional, Sequence, Union

import flax.linen as nn
import jax.numpy as jnp

from jaxrl_m.common.initialization import init_fns


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Union[Callable[[jnp.ndarray], jnp.ndarray], str] = nn.swish
    activate_final: bool = False
    use_layer_norm: bool = False
    use_group_norm: bool = False
    dropout_rate: Optional[float] = None
    kernel_init_type: Optional[str] = None
    kernel_scale_final: Optional[float] = None
    use_feature_normalization: bool = False

    def setup(self):
        assert not (self.use_layer_norm and self.use_group_norm)
        self.init_fn = init_fns[self.kernel_init_type]

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        activations = self.activations
        if isinstance(activations, str):
            activations = getattr(nn, activations)

        for i, size in enumerate(self.hidden_dims):
            # optinally final layer have different init scale
            if i + 1 == len(self.hidden_dims) and self.kernel_scale_final is not None:
                x = nn.Dense(size, kernel_init=self.init_fn(self.kernel_scale_final))(x)
            else:
                x = nn.Dense(size, kernel_init=self.init_fn())(x)

            # normalization and activation after each layer
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                elif self.use_group_norm:
                    x = nn.GroupNorm()(x)
                x = activations(x)

        if self.use_feature_normalization:
            # Divide x by its L2 norm
            denominator = jnp.linalg.norm(x, axis=-1, keepdims=True)
            x = x / jnp.maximum(denominator, 1e-8)
        return x


class LayerInputMLP(MLP):
    """
    MLP, but each layer takes in an additional input as well
    such as the critic network in PTR
    """

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, layer_input: jnp.ndarray, train: bool = False
    ) -> jnp.ndarray:
        assert isinstance(layer_input, jnp.ndarray)
        activations = self.activations
        if isinstance(activations, str):
            activations = getattr(nn, activations)

        for i, size in enumerate(self.hidden_dims):
            x = jnp.concatenate([x, layer_input], axis=-1)  # difference from MLP

            # optinally final layer have different init scale
            if i + 1 == len(self.hidden_dims) and self.kernel_scale_final is not None:
                x = nn.Dense(size, kernel_init=self.init_fn(self.kernel_scale_final))(x)
            else:
                x = nn.Dense(size, kernel_init=self.init_fn())(x)

            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                elif self.use_group_norm:
                    x = nn.GroupNorm()(x)
                x = activations(x)
        return x


class MLPResNetBlock(nn.Module):
    features: int
    act: Callable
    dropout_rate: float = None
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, x, train: bool = False):
        residual = x
        if self.dropout_rate is not None and self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        x = nn.Dense(self.features * 4)(x)
        x = self.act(x)
        x = nn.Dense(self.features)(x)

        if residual.shape != x.shape:
            residual = nn.Dense(self.features)(residual)

        return residual + x


class MLPResNet(nn.Module):
    num_blocks: int
    out_dim: int
    dropout_rate: float = None
    use_layer_norm: bool = False
    hidden_dim: int = 256
    activations: Callable = nn.swish
    kernel_init_type: Optional[str] = None

    def setup(self):
        self.init_fn = init_fns[self.kernel_init_type]

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim, kernel_init=self.init_fn())(x)
        for _ in range(self.num_blocks):
            x = MLPResNetBlock(
                self.hidden_dim,
                act=self.activations,
                use_layer_norm=self.use_layer_norm,
                dropout_rate=self.dropout_rate,
            )(x, train=train)

        x = self.activations(x)
        x = nn.Dense(self.out_dim, kernel_init=self.init_fn())(x)
        return x


class Scalar(nn.Module):
    init_value: float

    def setup(self):
        self.value = self.param("value", lambda x: self.init_value)

    def __call__(self):
        return self.value
