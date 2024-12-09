from functools import partial
from typing import Any, Dict, Optional
from dataclasses import field
import distrax
import flax.linen as nn
import jax.numpy as jnp

from jaxrl_m.common.initialization import init_fns
from jaxrl_m.networks.distributional import hl_gauss_transform
from jaxrl_m.networks.mlp import Scalar


class ValueCritic(nn.Module):
    encoder: Optional[nn.Module]
    network: nn.Module
    init_final: Optional[float] = None
    kernel_init_type: Optional[str] = None

    def setup(self):
        self.init_fn = init_fns[self.kernel_init_type]

    @nn.compact
    def __call__(self, observations: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        if self.encoder is None:
            obs_enc = observations
        else:
            obs_enc = self.encoder(observations, train=train)
        outputs = self.network(obs_enc, train)
        if self.init_final is not None:
            value = nn.Dense(
                1,
                kernel_init=nn.initializers.uniform(-self.init_final, self.init_final),
            )(outputs)
        else:
            value = nn.Dense(1, kernel_init=self.init_fn())(outputs)
        return jnp.squeeze(value, -1)


class Critic(nn.Module):
    encoder: Optional[nn.Module]
    network: nn.Module
    init_final: Optional[float] = None
    network_separate_action_input: bool = (
        False  # input action to every layer (e.g. PTR)
    )
    kernel_init_type: Optional[str] = None
    kernel_init_params: Optional[Dict[str, Any]] = field(default_factory=dict)

    def setup(self):
        self.init_fn = init_fns[self.kernel_init_type]

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        train: bool = False,
    ) -> jnp.ndarray:
        if self.encoder is None:
            obs_enc = observations
        else:
            obs_enc = self.encoder(observations, train=train)

        if self.network_separate_action_input:
            outputs = self.network(obs_enc, actions, train)
        else:
            inputs = jnp.concatenate([obs_enc, actions], -1)
            outputs = self.network(inputs, train)
        if self.init_final is not None:
            value = nn.Dense(
                1,
                kernel_init=nn.initializers.uniform(-self.init_final, self.init_final),
            )(outputs)
        else:
            value = nn.Dense(1, kernel_init=self.init_fn(**self.kernel_init_params))(
                outputs
            )
        return jnp.squeeze(value, -1)


class DistributionalCritic(nn.Module):
    encoder: Optional[nn.Module]
    network: nn.Module
    q_low: float
    q_high: float
    num_bins: int = 51
    init_final: Optional[float] = None
    network_separate_action_input: bool = (
        False  # input action to every layer (e.g. PTR)
    )
    kernel_init_type: Optional[str] = None
    kernel_init_params: Optional[Dict[str, Any]] = field(default_factory=dict)

    def setup(self) -> None:
        self.init_fn = init_fns[self.kernel_init_type]
        # set up the histogram loss transform
        self.target_to_probs, self.probs_to_target = hl_gauss_transform(
            self.q_low, self.q_high, self.num_bins
        )

        return super().setup()

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        train: bool = False,
    ) -> jnp.ndarray:
        if self.encoder is None:
            obs_enc = observations
        else:
            obs_enc = self.encoder(observations, train=train)

        if self.network_separate_action_input:
            outputs = self.network(obs_enc, actions, train)
        else:
            inputs = jnp.concatenate([obs_enc, actions], -1)
            outputs = self.network(inputs, train)
        if self.init_final is not None:
            logits = nn.Dense(
                self.num_bins,
                kernel_init=nn.initializers.uniform(-self.init_final, self.init_final),
            )(outputs)
        else:
            logits = nn.Dense(
                self.num_bins, kernel_init=self.init_fn(**self.kernel_init_params)
            )(outputs)

        # apply softmax
        probs = nn.softmax(logits, axis=-1)
        probs_mean = self.probs_to_target(probs)

        return probs_mean, logits


class ContrastiveCritic(nn.Module):
    encoder: nn.Module
    sa_net: nn.Module
    g_net: nn.Module
    repr_dim: int = 16
    twin_q: bool = True
    sa_net2: Optional[nn.Module] = None
    g_net2: Optional[nn.Module] = None
    init_final: Optional[float] = None

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, train: bool = False
    ) -> jnp.ndarray:
        obs_goal_encoding = self.encoder(observations)
        encoding_dim = obs_goal_encoding.shape[-1] // 2
        obs_encoding, goal_encoding = (
            obs_goal_encoding[..., :encoding_dim],
            obs_goal_encoding[..., encoding_dim:],
        )

        if self.init_final is not None:
            kernel_init = partial(
                nn.initializers.uniform, -self.init_final, self.init_final
            )
        else:
            kernel_init = default_init

        sa_inputs = jnp.concatenate([obs_encoding, actions], -1)
        sa_repr = self.sa_net(sa_inputs, train=train)
        sa_repr = nn.Dense(self.repr_dim, kernel_init=kernel_init())(sa_repr)
        g_repr = self.g_net(goal_encoding, train=train)
        g_repr = nn.Dense(self.repr_dim, kernel_init=kernel_init())(g_repr)
        outer = jnp.einsum("ik,jk->ij", sa_repr, g_repr)

        if self.twin_q:
            sa_repr2 = self.sa_net2(sa_inputs, train=train)
            sa_repr2 = nn.Dense(self.repr_dim, kernel_init=kernel_init())(sa_repr2)
            g_repr2 = self.g_net2(goal_encoding, train=train)
            g_repr2 = nn.Dense(self.repr_dim, kernel_init=kernel_init())(g_repr2)
            outer2 = jnp.einsum("ik,jk->ij", sa_repr2, g_repr2)

            outer = jnp.stack([outer, outer2], axis=-1)

        return outer


def ensemblize(cls, num_qs, out_axes=0):
    return nn.vmap(
        cls,
        variable_axes={"params": 0},
        split_rngs={"params": True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_qs,
    )


class Policy(nn.Module):
    encoder: Optional[nn.Module]
    network: nn.Module
    action_dim: int
    init_final: Optional[float] = None
    std_parameterization: str = "exp"  # "exp", "softplus", "fixed", or "uniform"
    std_min: Optional[float] = 1e-5
    std_max: Optional[float] = 10.0
    log_std_multiplier: float = 1.0
    log_std_offset: float = -1.0
    tanh_squash_distribution: bool = False
    fixed_std: Optional[jnp.ndarray] = None
    kernel_init_type: Optional[str] = None
    kernel_init_params: Optional[Dict[str, Any]] = field(default_factory=dict)

    def setup(self):
        self.init_fn = init_fns[self.kernel_init_type]

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, temperature: float = 1.0, train: bool = False
    ) -> distrax.Distribution:
        if self.encoder is None:
            obs_enc = observations
        else:
            obs_enc = self.encoder(observations, train=train)

        outputs = self.network(obs_enc, train)

        means = nn.Dense(
            self.action_dim, kernel_init=self.init_fn(**self.kernel_init_params)
        )(outputs)
        if self.fixed_std is None:
            if self.std_parameterization == "exp":
                log_stds = nn.Dense(
                    self.action_dim,
                    kernel_init=self.init_fn(**self.kernel_init_params),
                )(outputs)
                log_stds = (
                    Scalar(self.log_std_multiplier)() * log_stds
                    + Scalar(self.log_std_offset)()
                )
                stds = jnp.exp(log_stds)
            elif self.std_parameterization == "softplus":
                stds = nn.Dense(
                    self.action_dim, kernel_init=self.init_fn(**self.kernel_init_params)
                )(outputs)
                stds = nn.softplus(stds)
            elif self.std_parameterization == "uniform":
                log_stds = self.param(
                    "log_stds", nn.initializers.zeros, (self.action_dim,)
                )
                stds = jnp.exp(log_stds)
            else:
                raise ValueError(
                    f"Invalid std_parameterization: {self.std_parameterization}"
                )
        else:
            assert self.std_parameterization == "fixed"
            stds = jnp.array(self.fixed_std)

        # Clip stds to avoid numerical instability
        # For a normal distribution under MaxEnt, optimal std scales with sqrt(temperature)
        stds = jnp.clip(stds, self.std_min, self.std_max) * jnp.sqrt(temperature)

        if self.tanh_squash_distribution:
            distribution = TanhMultivariateNormalDiag(
                loc=means,
                scale_diag=stds,
            )
        else:
            distribution = distrax.MultivariateNormalDiag(
                loc=means,
                scale_diag=stds,
            )

        return distribution


class TanhMultivariateNormalDiag(distrax.Transformed):
    def __init__(
        self,
        loc: jnp.ndarray,
        scale_diag: jnp.ndarray,
        low: Optional[jnp.ndarray] = None,
        high: Optional[jnp.ndarray] = None,
    ):
        distribution = distrax.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)

        layers = []

        if not (low is None or high is None):

            def rescale_from_tanh(x):
                x = (x + 1) / 2  # (-1, 1) => (0, 1)
                return x * (high - low) + low

            def forward_log_det_jacobian(x):
                high_ = jnp.broadcast_to(high, x.shape)
                low_ = jnp.broadcast_to(low, x.shape)
                return jnp.sum(jnp.log(0.5 * (high_ - low_)), -1)

            layers.append(
                distrax.Lambda(
                    rescale_from_tanh,
                    forward_log_det_jacobian=forward_log_det_jacobian,
                    event_ndims_in=1,
                    event_ndims_out=1,
                )
            )

        layers.append(distrax.Block(distrax.Tanh(), 1))

        bijector = distrax.Chain(layers)

        super().__init__(distribution=distribution, bijector=bijector)

    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())

    def stddev(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.stddev())
