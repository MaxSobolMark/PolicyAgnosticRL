import jax
import jax.numpy as jnp
from flax import linen as nn

from .models.infrastructure import MLP
from .models.q_transformer import QTConfig, QTransformer
from .utils import Tokenizer


class TransformerQFunction:
    def __init__(
        self,
        bt_config: dict,
        mlp_config: dict,
        o_tokenizer_config: dict = None,
        a_tokenizer_config: dict = None,
        distributional_q: bool = False,
    ):

        # Save the configs
        if o_tokenizer_config is not None and a_tokenizer_config is not None:
            self._o_tokenizer_config = o_tokenizer_config
            self._a_tokenizer_config = a_tokenizer_config
        else:
            print("Will not use tokenizers inside the QFunction calls")
        self._bt_config = QTConfig(**bt_config)  # Adds additional required arguments
        self._mlp_config = mlp_config

        if hasattr(self, "_o_tokenizer_config") and hasattr(
            self, "_a_tokenizer_config"
        ):
            self.o_tokenizer = Tokenizer(**self._o_tokenizer_config)
            self.a_tokenizer = Tokenizer(**self._a_tokenizer_config)

        self.bt = QTransformer(self._bt_config)

        mlp_op_multiplier = self._mlp_config["num_bins"] if distributional_q else 1
        self.action_tokens_func = MLP(
            # self._bt_config.a_vocab_size * mlp_op_multiplier,
            1 * mlp_op_multiplier,
            hidden_size=self._mlp_config["hidden_size"],
        )
        self.eos_func = MLP(
            mlp_op_multiplier, hidden_size=self._mlp_config["hidden_size"]
        )

        self.distributional_q = distributional_q

        if self.distributional_q:
            self._support = jnp.linspace(
                self._mlp_config["min_value"],
                self._mlp_config["max_value"],
                # self._mlp_config["num_bins"] + 1,
                self._mlp_config["num_bins"],
                dtype=jnp.float32,
            )

    def from_dist(self, x, apply_softmax=True):
        """Computes the expectation of the distribution.

        Args:
            x: (*batch_dims, num_bins)
        """
        if not self.distributional_q:
            raise ValueError("Not a distributional Q function")

        centers = (self._support[:-1] + self._support[1:]) / 2  # (num_bins,)
        if apply_softmax:
            probs = jax.nn.softmax(x, axis=-1)
        else:
            probs = x
        return jnp.sum(probs * centers, axis=-1)  # (*batch_dims)

    def make_dist(self, x, num_bins, apply_softmax=False):
        """Creates a distribution from logits.

        Args:
            x: (*batch_dims, num_bins)
        """
        if not self.distributional_q:
            raise ValueError("Not a distributional Q function")
        x_shape = x.shape
        batch_dims = x_shape[:-1]
        assert x_shape[-1] % num_bins == 0
        op_dim = int(x_shape[-1] / num_bins)
        probs_shape = batch_dims + (op_dim, num_bins)
        logits = x.reshape(*probs_shape)
        if apply_softmax:
            probs = jax.nn.softmax(logits, axis=-1)
            return probs
        else:
            return logits

    def get_new_params(self, rngs):
        action_tokens_func_params = self.action_tokens_func.init(
            dict(params=rngs["action_tokens_func"]),
            jnp.zeros((1, self._bt_config.hidden_size)),
        )
        eos_func_params = self.eos_func.init(
            dict(params=rngs["eos_func"]),
            jnp.zeros((1, self._bt_config.hidden_size)),
        )

        bt_params = self.bt.init_weights(
            dict(params=rngs["bt_params"], dropout=rngs["bt_dropout"]), (1, 1)
        )

        params = dict(
            action_tokens_func=action_tokens_func_params,
            eos_func=eos_func_params,
            bt=bt_params,
        )

        return params

    def __call__(
        self,
        rngs,
        params,
        o,
        a_discretized=None,
        deterministic=True,
        add_eos_token=False,
        get_q_dist=False,
        normalize=False,
    ):
        """(ip_observations, ip_actions): -> seperately (discretize -> embeddings) \
                                          -> bt -> mlp_head -> op Q values
        """
        # Unpack the params
        action_tokens_func_params = params["action_tokens_func"]
        eos_func_params = params["eos_func"]
        bt_params = params["bt"]

        # Pass throught the transformer
        bt_op = self.bt(
            bt_params,
            # o_discretized,
            o,
            a_discretized,
            # dropout_rng=rngs['bt'],
            rngs=dict(
                params=rngs["bt_params"],
                dropout=rngs["bt_dropout"],
            ),
            train=not deterministic,
            add_eos_token=add_eos_token,
        )

        op_tokens = bt_op.last_hidden_state[:, :, :]

        if normalize:
            norms = jnp.linalg.norm(op_tokens, axis=-1, keepdims=True)
            norms = jnp.maximum(norms, 1e-6)
            op_tokens = op_tokens / norms

        if add_eos_token:
            # (batch_size, n_S + n_A + 1, vocab_size) if not self.distributional_q
            # (batch_size, n_S + n_A + 1, vocab_size * num_bins) otherwise
            action_func_op = self.action_tokens_func.apply(
                action_tokens_func_params,
                op_tokens[:, :-1, :],
                rngs=dict(params=rngs["action_tokens_func"]),
            )
            # (batch_size, 1, 1) if not self.distributional_q
            # (batch_size, 1, num_bins) otherwise
            eos_func_op = self.eos_func.apply(
                eos_func_params,
                op_tokens[:, -1:, :],
                rngs=dict(params=rngs["eos_func"]),
            )

            # (batch_size, n_S + n_A, vocab_size), (batch_size, 1, 1)
            if self.distributional_q:
                q_dist = self.make_dist(
                    action_func_op,
                    num_bins=self._mlp_config["num_bins"],
                    apply_softmax=False,
                )  # (*batch_dims, seq_len, num_bins)
                # q_values = self.from_dist(
                #     q_dist, apply_softmax=True
                # )  # (*batch_dims, seq_len)
                q_values = None

                eos_q_dist = self.make_dist(
                    eos_func_op,
                    num_bins=self._mlp_config["num_bins"],
                    apply_softmax=False,
                )  # (*batch_dims, 1, num_bins)
                eos_q_values = self.from_dist(
                    eos_q_dist, apply_softmax=True
                )  # (*batch_dims, 1)
            else:
                q_values = action_func_op
                eos_q_values = eos_func_op

        else:
            # (batch_size, *, vocab_size) if not self.distributional_q
            # (batch_size, *, vocab_size * num_bins) otherwise
            action_func_op = self.action_tokens_func.apply(
                action_tokens_func_params,
                op_tokens,
                rngs=dict(params=rngs["action_tokens_func"]),
            )

            # (batch_size, n_S + n_A, vocab_size), (batch_size, 1, 1)
            if self.distributional_q:
                q_dist = self.make_dist(
                    action_func_op,
                    num_bins=self._mlp_config["num_bins"],
                    apply_softmax=False,
                )  # (*batch_dims, seq_len, num_bins)
                # q_values = self.from_dist(
                #     q_dist, apply_softmax=True
                # )
                # (*batch_dims, seq_len)
                q_values = None
            else:
                q_values = action_func_op
            eos_q_values = None
            eos_q_dist = None

        if get_q_dist:
            assert (
                self.distributional_q
            ), "Cannot get q_dist for non-distributional Q function"
            return q_values, eos_q_values, q_dist, eos_q_dist
        else:
            return q_values, eos_q_values

    @nn.nowrap
    def rng_keys(self):
        return ("bt_dropout", "bt_params", "action_tokens_func", "eos_func")
