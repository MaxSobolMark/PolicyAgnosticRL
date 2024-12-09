import jax
import jax.numpy as jnp

class Tokenizer:
    @staticmethod
    def quantize(ip: jnp.ndarray,
                 low: float = -1,
                 high: float = 1,
                 num_bins: int = 10) -> jnp.ndarray:
        assert low.shape[-1] == high.shape[-1] == ip.shape[-1]
        def quantize_single_dimension(x, low, high, num_bins): # TODO: vectorize this
            q_state = []
            for j in range(low.shape[0]):
                bins = jnp.linspace(low[j], high[j], num_bins + 1)
                q = jnp.digitize(x[j], bins=bins) - 1
                q = jnp.clip(q, 0, num_bins - 1) # Very important otherwise, sometimes -1, num_bins are returned
                q_state.append(q)
            return jnp.array(q_state)
        quantized_ip = jax.vmap(quantize_single_dimension, in_axes=(0, None, None, None))(ip, low, high, num_bins)
        return quantized_ip

    def __init__(self, **kwargs):
        self.low = kwargs.get('low')
        self.high = kwargs.get('high')
        self.num_bins = kwargs.get('num_bins')

    def __call__(self, ip: jnp.ndarray):
        """ip: (batch_size, n_dims)"""
        digitized = self.quantize(ip, self.low, self.high, self.num_bins) # (batch_size, n_dims)
        # tokens = dict(input_ids=digitized, attention_mask=jnp.ones_like(digitized))
        # tokens = dict(input_ids=digitized)
        # return tokens
        return digitized