from typing import Optional

import jax
import jax.numpy as jnp


def _log1mexp(x: jax.Array) -> jax.Array:
    """Compute `log(1 - exp(- |x|))` elementwise in a numerically stable way.

    Source:
    https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/math/generic.py#L685-L709

    Args:
    x: float value.

    Returns:
    `log(1 - exp(- |x|))`
    """
    x = jnp.abs(x)
    return jnp.where(
        x < jnp.log(2),
        jnp.log(-jnp.expm1(-x)),
        jnp.log1p(-jnp.exp(-x)),
    )


def _log_sub_exp(x: jax.Array, y: jax.Array) -> jax.Array:
    """Compute `log(exp(max(x, y)) - exp(min(x, y)))` in a numerically stable way.

    Source:
    https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/math/generic.py#L653-L682

    Args:
    x: float value.
    y: float value.

    Returns:
    `log(exp(max(x, y)) - exp(min(x, y)))`
    """
    larger = jnp.maximum(x, y)
    smaller = jnp.minimum(x, y)
    return larger + _log1mexp(jnp.maximum(larger - smaller, 0))


def _normal_cdf_log_difference(x: jax.Array, y: jax.Array) -> jax.Array:
    """Computes `log(ndtr(x) - ndtr(y)) assuming that x >= y.

    Source:
    https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/distributions/truncated_normal.py#L61-L69

    Args:
    x: larger query point.
    y: smaller query point.

    Returns:
    `log(ndtr(x) - ndtr(y))`
    """
    # When x >= y >= 0 we will return log(ndtr(-y) - ndtr(-x))
    # because ndtr does not have good precision for large positive x, y.
    is_y_positive = y >= 0
    x_hat = jnp.where(is_y_positive, -y, x)
    y_hat = jnp.where(is_y_positive, -x, y)
    return _log_sub_exp(
        jax.scipy.special.log_ndtr(x_hat),
        jax.scipy.special.log_ndtr(y_hat),
    )


def hl_gauss_transform(
    min_value: float,
    max_value: float,
    num_bins: int,
    sigma: Optional[float] = None,
):
    """
    HL-Gauss: A numerically stable histogram loss transform for a truncated normal.
    This is used for the cross-entropy loss for a distributional critic.

    Usuage: forward_fn = lambda logits, target: cross_entropy(
        logits, self.transform_to_probs(target)
    )
    """
    if sigma is None:
        # set to default value suggested by the paper
        # https://arxiv.org/pdf/2403.03950.pdf
        sigma = 0.75 * (max_value - min_value) / num_bins
    support = jnp.linspace(min_value, max_value, num_bins + 1, dtype=jnp.float32)

    def transform_to_probs(target: jax.Array) -> jax.Array:
        bin_log_probs = _normal_cdf_log_difference(
            (support[1:] - target) / (jnp.sqrt(2) * sigma),
            (support[:-1] - target) / (jnp.sqrt(2) * sigma),
        )
        log_z = _normal_cdf_log_difference(
            (support[-1] - target) / (jnp.sqrt(2) * sigma),
            (support[0] - target) / (jnp.sqrt(2) * sigma),
        )
        return jnp.exp(bin_log_probs - log_z)

    def transform_from_probs(probs: jax.Array) -> jax.Array:
        centers = (support[:-1] + support[1:]) / 2
        return jnp.sum(probs * centers, axis=-1)

    return transform_to_probs, transform_from_probs


def cross_entropy_loss_on_scalar(
    logits: jax.Array, target: jax.Array, target_to_dist_fn: jax.Array
) -> jax.Array:
    """
    Compute the cross-entropy loss between the logits and the target distribution.
    Turn the scalar target into a distributio with target_to_dist_fn.

    logits: (n_ensemble, batch_size, n_bins)
    target: (batch_size,)
    """
    target_probs = jax.vmap(target_to_dist_fn, in_axes=0, out_axes=0,)(
        target
    )  # map over batch dimension
    return -jnp.sum(target_probs * jax.nn.log_softmax(logits, axis=-1), axis=-1)
