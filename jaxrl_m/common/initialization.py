from typing import Optional

import flax.linen as nn
import jax.numpy as jnp


def var_scaling_init(scale: Optional[float] = 1.0):
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


def orthogonal_init(scale: Optional[float] = jnp.sqrt(2.0)):
    return nn.initializers.orthogonal(scale)


def xavier_normal_init():
    return nn.initializers.xavier_normal()


def kaiming_init():
    return nn.initializers.kaiming_normal()


def xavier_uniform_init():
    return nn.initializers.xavier_uniform()


init_fns = {
    None: orthogonal_init,
    "var_scaling": var_scaling_init,
    "orthogonal": orthogonal_init,
    "xavier_normal": xavier_normal_init,
    "kaiming": kaiming_init,
    "xavier_uniform": xavier_uniform_init,
}
