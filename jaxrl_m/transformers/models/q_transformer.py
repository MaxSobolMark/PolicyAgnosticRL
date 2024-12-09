from typing import Any, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import transformers
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from transformers.modeling_flax_outputs import (
    FlaxBaseModelOutputWithPastAndCrossAttentions,
)
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_flax_gpt2 import FlaxGPT2BlockCollection


class MLP(nn.Module):
    out_dims: int
    hidden_size: int = 128

    @nn.compact
    def __call__(self, x):
        *batch_dims, n_ip_dims = x.shape
        x = x.reshape((-1, n_ip_dims))
        x = nn.Dense(self.hidden_size)(x)
        x = nn.elu(x)
        x = nn.Dense(self.out_dims)(x)
        return x.reshape((*batch_dims, self.out_dims))


class QTConfig(GPT2Config):
    model_type = "gpt2"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        a_vocab_size=int(2**10),
        o_vocab_size=int(2**10),
        n_layer=2,
        n_head=12,
        hidden_size=768,
        causal=True,
        add_eos_token=False,
        n_A=None,
        n_S=None,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.a_vocab_size = a_vocab_size
        self.o_vocab_size = o_vocab_size
        self.n_A = n_A
        self.n_S = n_S
        self.n_layer = n_layer
        # self.num_hidden_layers = num_hidden_layers
        self.n_head = n_head
        self.hidden_size = hidden_size
        self.causal = causal
        self.add_eos_token = add_eos_token


class IndexEmbed(nn.Module):
    vocab_size: int
    n_index: int
    embed_dim: int
    dtype: jnp.float32
    init_range: float

    def setup(self):
        self.embeds = [
            nn.Embed(
                self.vocab_size,
                self.embed_dim,
                embedding_init=jax.nn.initializers.normal(stddev=self.init_range),
                dtype=self.dtype,
            )
            for _ in range(self.n_index)
        ]

    def __call__(self, input_ids):
        # Fetch the embeddings for corresponding indexes
        def apply_embed(embeds, ind, input_ids):
            return embeds[ind](input_ids[:, ind : ind + 1])

        op = []
        batch_size, n_dims = input_ids.shape
        for i in range(n_dims):
            op.append(apply_embed(self.embeds, i, input_ids))

        op = jnp.concatenate(op, axis=1)
        return op


class QTransformerModule(nn.Module):
    """
    References:
        https://github.com/huggingface/transformers/blob/73a27345d47a5a6e82f4104daffd3fa998668664/src/transformers/models/gpt2/modeling_flax_gpt2.py#L536-L537
    """

    config: QTConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embed_dim = self.config.hidden_size
        # if self.embed_dim % 2 != 0:
        #     raise ValueError("Hidden size should be a multiple of 2")
        # self.o_embed_dim = self.a_embed_dim = self.embed_dim // 2
        self.o_embed_dim = self.a_embed_dim = self.embed_dim

        # # Observation embeddings
        # self.o_embeddings = IndexEmbed(
        #     self.config.o_vocab_size,
        #     n_index=self.config.n_S,
        #     embed_dim=self.o_embed_dim,
        #     init_range=self.config.initializer_range,
        #     dtype=self.dtype
        # )
        self.o_mlp = MLP(self.o_embed_dim)

        # Action embeddings
        self.a_embeddings = IndexEmbed(
            self.config.a_vocab_size,
            n_index=self.config.n_A,
            embed_dim=self.a_embed_dim,
            init_range=self.config.initializer_range,
            dtype=self.dtype,
        )

        self.eos_embeddings = nn.Embed(
            1,
            self.a_embed_dim,
            embedding_init=jax.nn.initializers.normal(
                stddev=self.config.initializer_range
            ),
            dtype=self.dtype,
        )

        self.wpe = nn.Embed(
            self.config.max_position_embeddings,
            self.embed_dim,
            embedding_init=jax.nn.initializers.normal(
                stddev=self.config.initializer_range
            ),
            dtype=self.dtype,
        )

        self.dropout = nn.Dropout(rate=self.config.embd_pdrop)
        self.h = FlaxGPT2BlockCollection(self.config, dtype=self.dtype)
        self.ln_f = nn.LayerNorm(
            epsilon=self.config.layer_norm_epsilon, dtype=self.dtype
        )

    def __call__(
        self,
        # input_o_ids, # Observation ids
        raw_o,
        input_a_ids,  # Action ids
        attention_mask,
        position_ids,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        deterministic=True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        add_eos_token: bool = False,
    ):
        # Assume fist passing in observations then action tokens
        # input_o_embeds = self.o_embeddings(input_o_ids.astype("i4")) # (batch_size, n_S, o_embed_dim)
        # input_o_embeds = None
        input_o_embeds = self.o_mlp(raw_o)[:, None, :]  # (batch_size, 1, o_embed_dim)
        if input_a_ids is not None:
            input_a_embeds = self.a_embeddings(
                input_a_ids.astype("i4")
            )  # (batch_size, n_A, a_embed_dim)
            input_embeds = jax.numpy.concatenate(
                [input_o_embeds, input_a_embeds], axis=-2
            )
        else:
            input_embeds = input_o_embeds
        if add_eos_token:
            input_eos_embeds = self.eos_embeddings(
                jnp.zeros((input_o_embeds.shape[0], 1)).astype("i4")
            )
            input_embeds = jax.numpy.concatenate(
                [input_embeds, input_eos_embeds], axis=-2
            )

        position_embeds = self.wpe(position_ids.astype("i4"))

        hidden_states = input_embeds + position_embeds
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)

        outputs = self.h(
            hidden_states,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1],
            attentions=outputs[2],
            cross_attentions=outputs[3],
        )


class QTransformer(transformers.modeling_flax_utils.FlaxPreTrainedModel):

    config_class = QTConfig
    base_model_prefix = "transformer"
    module_class: nn.Module = QTransformerModule

    def __init__(
        self,
        config: GPT2Config,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(
            config,
            module,
            input_shape=input_shape,
            seed=seed,
            dtype=dtype,
            _do_init=_do_init,
        )

    def init_weights(
        self, rngs: jax.random.PRNGKey, input_shape, params: FrozenDict = None
    ) -> FrozenDict:
        # init input tensors
        # input_o_ids = jnp.zeros((1, self.config.n_S), dtype="i4")
        input_o_raw = jnp.zeros((1, self.config.n_S))
        input_a_ids = jnp.zeros((1, self.config.n_A), dtype="i4")

        seq_len = 1 + input_a_ids.shape[-1]
        if self.config.add_eos_token:
            seq_len += 1
        seq_shape = input_a_ids.shape[:-1] + (seq_len,)

        attention_mask = jnp.ones(seq_shape, dtype=input_o_raw.dtype)
        position_ids = jnp.broadcast_to(jnp.arange(seq_len), seq_shape)

        # params_rng, dropout_rng = jax.random.split(rng)
        # rngs = {"params": params_rng, "dropout": dropout_rng}

        if self.config.add_cross_attention:
            raise NotImplementedError
        else:
            module_init_outputs = self.module.init(
                rngs,
                input_o_raw,
                input_a_ids,
                attention_mask,
                position_ids,
                return_dict=False,
                add_eos_token=self.config.add_eos_token,
            )

        random_params = module_init_outputs["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
        """
        raise NotImplementedError
        # init input variables to retrieve cache
        input_o_ids = jnp.ones((batch_size, max_length))
        input_a_ids = jnp.ones((batch_size, max_length))

        input_ids = jnp.concatenate([input_o_ids, input_a_ids], axis=-1)

        attention_mask = jnp.ones_like(input_o_ids)
        position_ids = jnp.broadcast_to(
            jnp.arange(jnp.atleast_2d(input_o_ids).shape[-1]), input_ids.shape
        )

        init_variables = self.module.init(
            jax.random.PRNGKey(0),
            input_o_ids,
            input_a_ids,
            attention_mask,
            position_ids,
            return_dict=False,
            init_cache=True,
        )
        return unfreeze(init_variables["cache"])

    def __call__(
        self,
        params: dict,
        input_o_raw,  # (batch_size, n_S)
        input_a_ids,  #  (batch_size, n_a_dimensions, a_id_dim) / (a_id_dim will most likely be 1)
        attention_mask=None,
        position_ids=None,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        past_key_values: dict = None,
        rngs: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        add_eos_token: bool = False,
    ):

        # assert input_o_ids.shape[0] == input_a_ids.shape[0]
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        if encoder_hidden_states is not None and encoder_attention_mask is None:
            batch_size, sequence_length = encoder_hidden_states.shape[:2]
            encoder_attention_mask = jnp.ones((batch_size, sequence_length))

        # batch_size, o_seq_len = input_o_ids.shape
        batch_size, *_ = input_o_raw.shape
        o_seq_len = 1
        if input_a_ids is not None:
            _, a_seq_len = input_a_ids.shape
        else:
            a_seq_len = 0

        sequence_length = a_seq_len + o_seq_len
        if add_eos_token:
            sequence_length += 1

        if position_ids is None:
            if past_key_values is not None:
                raise ValueError(
                    "Make sure to provide `position_ids` when passing `past_key_values`."
                )

            position_ids = jnp.broadcast_to(
                jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
            )

        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        # Handle any PRNG if needed
        # rngs = dict()
        # # if dropout_rng is not None:
        #     rngs["dropout"] = dropout_rng

        inputs = dict(params=params)

        # if past_key_values are passed then cache is already initialized a private flag init_cache has to be passed down to ensure cache is used. It has to be made sure that cache is marked as mutable so that it can be changed by FlaxGPT2Attention module
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        outputs = self.module.apply(
            inputs,
            input_o_raw,
            # jnp.array(input_o_ids, dtype="i4"),
            jnp.array(input_a_ids, dtype="i4") if input_a_ids is not None else None,
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            encoder_hidden_states,
            encoder_attention_mask,
            not train,
            False,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
            mutable=mutable,
            add_eos_token=add_eos_token,
        )

        # add updated cache to model output
        if past_key_values is not None and return_dict:
            outputs, past_key_values = outputs
            outputs["past_key_values"] = unfreeze(past_key_values["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past_key_values = outputs
            outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

        return outputs


if __name__ == "__main__":

    key = 0

    config = QTConfig()
    model = QTransformer(config)

    rng_key = jax.random.PRNGKey(key)

    model_params = model.init_weights(rng_key, (1, 1))

    input_o_ids = jnp.ones((1, 1), dtype=jnp.int32)
    input_a_ids = jnp.ones((1, 1), dtype=jnp.int32)

    inputs = (input_o_ids, input_a_ids)

    ops = model(model_params, *inputs)
    import ipdb

    ipdb.set_trace()
    print(ops)
