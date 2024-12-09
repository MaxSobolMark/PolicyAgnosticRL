import os
import shutil
from functools import partial
from typing import Optional

import gym
import chex
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import orbax
from flax.training import orbax_utils
from jaxrl_m.common.common import JaxRLTrainState
from jaxrl_m.common.typing import Batch, PRNGKey
from jaxrl_m.vision.data_augmentations import batched_random_crop
from jaxrl_m.transformers.q_functions import TransformerQFunction


class AutoRegressiveTransformerAgent:

    def forward_policy_autoregressive_inference(
        self, rng: PRNGKey, params, observations, sample: bool
    ):
        rng, bt_dropout_key, bt_params_key, action_tokens_func_key, eos_func_key = (
            jax.random.split(rng, 5)
        )
        keys = {
            "bt_dropout": bt_dropout_key,
            "bt_params": bt_params_key,
            "action_tokens_func": action_tokens_func_key,
            "eos_func": eos_func_key,
        }
        action_tokens = None

        for current_dim in range(self.config["action_dim"]):
            all_action_dimensions_distribution = self.transformer(
                keys,
                params,
                observations,
                action_tokens,
                normalize=False,
                deterministic=False,
                get_q_dist=True,
                add_eos_token=False,
            )[2]
            chex.assert_shape(
                all_action_dimensions_distribution,
                (
                    observations.shape[0],
                    # 1,
                    current_dim + 1,
                    1,  # originally this was vocab size for Q-function
                    self.config["num_bins"],
                ),
            )

            # next_action_dimension = all_action_dimensions[:, :, -1, :]
            next_action_dimension_distribution = all_action_dimensions_distribution[
                :, -1, :
            ]

            if sample:
                next_action_dimension_token = jax.random.categorical(
                    rng, logits=next_action_dimension_distribution, axis=-1
                )
            else:
                next_action_dimension_token = jnp.argmax(
                    next_action_dimension_distribution, axis=-1
                )
            chex.assert_shape(
                # next_action_dimension_token, (observations.shape[0], 1, 1)
                next_action_dimension_token,
                (observations.shape[0], 1),
            )

            if action_tokens is None:
                action_tokens = next_action_dimension_token
            else:
                action_tokens = jnp.concatenate(
                    [action_tokens, next_action_dimension_token], axis=-1
                )

            chex.assert_shape(
                # action_tokens, (observations.shape[0], 1, current_dim + 1)
                action_tokens,
                (observations.shape[0], current_dim + 1),
            )

        # action_tokens = jnp.squeeze(action_tokens, axis=1)
        chex.assert_shape(
            action_tokens, (observations.shape[0], self.config["action_dim"])
        )

        return action_tokens

    def sample_actions(self, *args, timer=None, **kwargs):
        actions = self._sample_actions(*args, self.state, **kwargs)
        return actions

    @partial(jax.jit, static_argnames=("self", "argmax", "repeat"))
    def _sample_actions(
        self,
        observations: jnp.ndarray,
        state,
        *,
        seed: PRNGKey = None,
        argmax: bool = False,
        repeat: int = 1,
        **kwargs,
    ) -> jnp.ndarray:
        """Sample actions from the model."""
        if isinstance(observations, dict):
            observations = observations["state"]
        need_to_unbatch = False
        if observations.ndim == 1:
            need_to_unbatch = True
            observations = observations[None]
        assert (
            observations.ndim == 2
            and observations.shape[1] == self.config["observation_dim"]
        ), observations.shape

        # Repeat observations along batch dimension
        batch_size = observations.shape[0]
        observations = (
            observations[:, None]
            .repeat(repeat, axis=1)
            .reshape(batch_size * repeat, observations.shape[1])
        )
        rng = state.rng if seed is None else seed
        rng, key = jax.random.split(rng)

        action_tokens = self.forward_policy_autoregressive_inference(
            key, state.params, observations, sample=not argmax
        )
        # self.transformer._support contains the quantile values (shape [num_bins])
        # action_tokens is the index of the quantile value in the support (shape [batch_size, action_dim])
        actions = jnp.take(self.transformer._support, action_tokens, axis=-1)
        chex.assert_shape(actions, (batch_size * repeat, self.config["action_dim"]))
        if need_to_unbatch and repeat == 1:
            return actions[0]
        return actions.reshape(batch_size, repeat, self.config["action_dim"])

    def update(self, batch: Batch, **kwargs):
        """Update the model parameters."""
        self.state, info = self._update(batch, self.state, **kwargs)
        return info

    @partial(jax.jit, static_argnames=("pmap_axis", "self"))
    def _update(self, batch: Batch, state, pmap_axis: str = None, **kwargs):
        """Update the model parameters."""
        if isinstance(batch["observations"], dict):
            batch["observations"] = batch["observations"]["state"]

        # Optionally apply DRQ augmentation
        rng = state.rng
        if self.config.get("drq_padding", 0) > 0:
            rng, key = jax.random.split(rng)
            # Use same key for both observations and next_observations
            batch["observations"]["image"] = batched_random_crop(
                batch["observations"]["image"],
                key,
                padding=self.config["drq_padding"],
                num_batch_dims=2,
            )
            batch["next_observations"]["image"] = batched_random_crop(
                batch["next_observations"]["image"],
                key,
                padding=self.config["drq_padding"],
                num_batch_dims=2,
            )

        def forward_policy_training(params, observations, action_tokens, rng):
            # Compute the log probabilities of the actions
            rng, bt_dropout_key, bt_params_key, action_tokens_func_key, eos_func_key = (
                jax.random.split(rng, 5)
            )
            keys = {
                "bt_dropout": bt_dropout_key,
                "bt_params": bt_params_key,
                "action_tokens_func": action_tokens_func_key,
                "eos_func": eos_func_key,
            }
            all_action_dimensions_distribution = self.transformer(
                keys,
                params,
                observations,
                action_tokens,
                normalize=False,
                deterministic=False,
                get_q_dist=True,
                add_eos_token=False,
            )[2]
            chex.assert_shape(
                all_action_dimensions_distribution,
                (
                    observations.shape[0],
                    self.config["action_dim"] + 1,
                    1,  # self.config["action_vocab_size"],
                    self.config["num_bins"],
                ),
            )
            all_action_dimensions_distribution = all_action_dimensions_distribution[
                :, :-1, 0, :
            ]

            all_action_dimensions_log_prob = jax.nn.log_softmax(
                all_action_dimensions_distribution, axis=-1
            )

            # Select the log probabilities of the actions
            action_tokens_log_prob = jnp.take_along_axis(
                all_action_dimensions_log_prob,
                action_tokens[..., None],
                axis=-1,
            )
            action_tokens_log_prob = jnp.squeeze(action_tokens_log_prob, axis=-1)
            chex.assert_shape(
                action_tokens_log_prob,
                (
                    observations.shape[0],
                    self.config["action_dim"],
                ),
            )

            return action_tokens_log_prob

        def actor_loss_fn(params, rng):
            observations = batch["observations"]
            if observations.ndim == 3:
                assert observations.shape[1] == 1 and batch["actions"].shape[1] == 1
                observations = observations[:, 0]
                batch["actions"] = batch["actions"][:, 0]
            assert (
                observations.ndim == 2
                and observations.shape[1] == self.config["observation_dim"]
            )
            target_actions = batch["actions"]
            assert (
                target_actions.ndim == 2
                and target_actions.shape[1] == self.config["action_dim"]
            )
            target_action_tokens = jnp.argmin(
                jnp.abs(
                    self.transformer._support[None, None] - target_actions[..., None]
                ),
                axis=-1,
            )
            chex.assert_shape(
                target_action_tokens,
                (observations.shape[0], self.config["action_dim"]),
            )

            rng, key = jax.random.split(rng)

            action_token_log_probs = forward_policy_training(
                params, observations, target_action_tokens, key
            )
            chex.assert_shape(
                action_token_log_probs,
                (observations.shape[0], self.config["action_dim"]),
            )

            # Compute the actor loss
            actor_loss = -jnp.mean(jnp.sum(action_token_log_probs, axis=-1))

            info = {
                "actor_loss": actor_loss,
            }
            return actor_loss, info

        loss_fns = {
            "actor": actor_loss_fn,
        }

        # compute gradients and update params
        new_state, info = state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )

        # update the target params
        new_state = new_state.target_update(self.config["target_update_rate"])
        new_state = new_state.replace(rng=rng)

        # log learning rates
        if self.lr_schedules["actor"] is not None:
            info["actor_lr"] = self.lr_schedules["actor"](state.step)

        # return self.replace(state=new_state), info
        return new_state, info

    def __init__(
        self,
        rng: PRNGKey,
        # example arrays for model init
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        action_space_low: jnp.ndarray,
        action_space_high: jnp.ndarray,
        # agent config
        encoder_def: nn.Module,
        # other shared network config
        image_observations: bool = True,
        # optimizer
        learning_rate: float = 3e-4,
        warmup_steps: int = 2000,
        actor_decay_steps: Optional[int] = None,
        # Transformer config
        transformer_num_layers: int = 4,
        transformer_hidden_size: int = 256,
        transformer_num_heads: int = 8,
        num_bins: int = 128,
        # action_vocab_size: int = (2**10),
        # observation_vocab_size: int = (2**9),
        target_update_rate=0.005,
        **kwargs,
    ):

        # Initialize the transformer
        assert not image_observations, "Only supports non-image observations for now"
        observation_dim = observations.shape[-1]
        action_dim = actions.shape[-1]

        transformer_config = {
            "a_vocab_size": num_bins,
            "n_layer": transformer_num_layers,
            "n_head": transformer_num_heads,
            "hidden_size": transformer_hidden_size,
            "add_eos_token": False,
            "n_S": observation_dim,
            "n_A": action_dim,
        }

        mlp_config = {
            "output_dim": action_dim,
            "hidden_size": transformer_hidden_size,
            "input_dim": transformer_hidden_size,
            "num_bins": num_bins,
            "min_value": action_space_low.min(),
            "max_value": action_space_high.max(),
        }

        transformer = TransformerQFunction(
            bt_config=transformer_config,
            mlp_config=mlp_config,
            distributional_q=True,
        )

        rng, bt_dropout_key, bt_params_key, action_tokens_func_key, eos_func_key = (
            jax.random.split(rng, 5)
        )

        params = jax.jit(transformer.get_new_params)(
            {
                "bt_dropout": bt_dropout_key,
                "bt_params": bt_params_key,
                "action_tokens_func": action_tokens_func_key,
                "eos_func": eos_func_key,
            },
        )

        # no decay
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=warmup_steps + 1,
            end_value=learning_rate,
        )
        lr_schedules = {
            "actor": lr_schedule,
        }
        if actor_decay_steps is not None:
            lr_schedules["actor"] = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=learning_rate,
                warmup_steps=warmup_steps,
                decay_steps=actor_decay_steps,
                end_value=0.0,
            )
        txs = {k: optax.adam(v) for k, v in lr_schedules.items()}

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=None,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )

        config = flax.core.FrozenDict(
            dict(
                image_observations=image_observations,
                observation_dim=observation_dim,
                action_dim=action_dim,
                num_bins=num_bins,
                target_update_rate=target_update_rate,
                **kwargs,
            )
        )
        self.transformer = transformer
        self.state = state
        self.config = config
        self.lr_schedules = lr_schedules

    def save_checkpoint(self, save_dir: str):
        checkpoint = {
            "state": self.state,
            "config": self.config,
            "lr_schedules": self.lr_schedules,
        }
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(checkpoint)
        os.makedirs(save_dir, exist_ok=True)

        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

        orbax_checkpointer.save(save_dir, checkpoint, save_args=save_args)

    def restore_checkpoint(self, restore_dir: str):
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint = orbax_checkpointer.restore(restore_dir)
        self.state = JaxRLTrainState.create(
            apply_fn=None,
            params=checkpoint["state"]["params"],
            txs=self.state.txs,
            target_params=checkpoint["state"]["target_params"],
            rng=checkpoint["state"]["rng"],
        )
        self.config = checkpoint["config"]
        self.lr_schedules = checkpoint["lr_schedules"]
