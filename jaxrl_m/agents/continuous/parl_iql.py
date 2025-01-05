import os
import shutil
from functools import partial
from typing import Optional

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import orbax
import tensorflow as tf
from flax.training import orbax_utils
from jaxrl_m.agents.continuous.action_optimization import (
    action_optimization_sample_actions,
)
from jaxrl_m.agents.continuous.iql import IQLAgent
from jaxrl_m.common.common import JaxRLTrainState
from jaxrl_m.common.typing import Data, Params, PRNGKey


# Using this so that jax knows which attributes are static.
@jax.tree_util.register_pytree_node_class
class PARLIQLAgent:
    def __getattr__(self, name):
        # Forward all unknown attributes to the IQLAgent class
        attribute = getattr(IQLAgent, name)

        if callable(attribute):
            return partial(attribute, self)
        return attribute

    def __init__(
        self,
        *args,
        num_base_policy_actions: int = 32,
        num_actions_to_keep: int = 10,
        num_steps: int = 10,
        step_size: float = 3e-4,
        optimize_critic_ensemble_min: bool = False,
        use_target_critic: bool = False,
        **kwargs,
    ):
        self.num_base_policy_actions = num_base_policy_actions
        self.num_actions_to_keep = num_actions_to_keep
        self.num_steps = num_steps
        self.step_size = step_size
        self.optimize_critic_ensemble_min = optimize_critic_ensemble_min
        self.use_target_critic = use_target_critic
        agent = super(PARLIQLAgent).create(*args, **kwargs)
        self.state = agent.state
        self.config = agent.config

    def tree_flatten(self):
        return (
            # Non-static:
            (self.state,),
            # Static:
            (
                self.num_base_policy_actions,
                self.num_actions_to_keep,
                self.num_steps,
                self.step_size,
                self.optimize_critic_ensemble_min,
                self.use_target_critic,
                dict(self.config),
            ),
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            num_base_policy_actions,
            num_actions_to_keep,
            num_steps,
            step_size,
            optimize_critic_ensemble_min,
            use_target_critic,
            config,
        ) = aux_data
        state = children[0]
        return cls(
            num_base_policy_actions=num_base_policy_actions,
            num_actions_to_keep=num_actions_to_keep,
            num_steps=num_steps,
            step_size=step_size,
            optimize_critic_ensemble_min=optimize_critic_ensemble_min,
            use_target_critic=use_target_critic,
            config=config,
            state=state,
        )

    def forward_policy(
        self,
        observations: Data,
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> distrax.Distribution:

        action_distribution, info_metrics = action_optimization_sample_actions(
            observations,
            critic_agent=self,
            critic_state=self.state,
            num_base_policy_actions=self.num_base_policy_actions,
            num_actions_to_keep=self.num_actions_to_keep,
            num_steps=self.num_steps,
            step_size=self.step_size,
            optimize_critic_ensemble_min=self.optimize_critic_ensemble_min,
            rng=rng,
            action_space_low=self.config["action_space_low"],
            action_space_high=self.config["action_space_high"],
            argmax=train,
        )

        return action_distribution

    def sample_actions(self, observations: Data, timer=None, **kwargs) -> jnp.ndarray:
        # timer is not hashable, so we take it out of the arguments
        return self._sample_actions(observations, **kwargs)

    @partial(jax.jit, static_argnames="argmax")
    def _sample_actions(
        self,
        observations: np.ndarray,
        *,
        seed: Optional[PRNGKey] = None,
        argmax=False,
        **kwargs,
    ) -> jnp.ndarray:
        if isinstance(observations, dict):
            key = "proprio" if "proprio" in observations else "state"
            if len(observations[key].shape) == 1:
                need_to_unbatch = True
                observations = jax.tree_map(
                    lambda x: jnp.expand_dims(x, 0), observations
                )
            else:
                need_to_unbatch = False
            assert len(observations[key].shape) == 2
        else:
            if len(observations.shape) == 1:
                need_to_unbatch = True
                observations = jnp.expand_dims(observations, 0)
            else:
                need_to_unbatch = False
            assert len(observations.shape) == 2, observations.shape
        dist = self.forward_policy(observations, rng=seed, train=False)
        if not argmax:
            action = dist.sample(seed=seed)
        else:
            # MixtureSameFamily doesn't implement mode, so first get most likely component
            # and then get the mode of that component
            most_likely_component = dist.mixture_distribution.probs.argmax(axis=-1)
            most_likely_actions = jnp.take_along_axis(
                dist.components_distribution.distribution.mode(),
                most_likely_component[:, None, None],
                axis=1,
            ).squeeze(axis=1)
            action = most_likely_actions

        if need_to_unbatch:
            assert len(action.shape) == 2 and action.shape[0] == 1
            action = action[0]

        return action

    def update(self, *args, **kwargs):
        # The IQL class has a gaussian policy by default, but PA-RL doesn't use it. Instead, we get
        # the actions from the base policy calculated outside (check train.py)
        new_agent, info = super().update(*args, **kwargs, train_actor=False)
        self.state = new_agent["state"]
        return info

    def to_device(self, sharding: jax.sharding.Sharding):
        self.state = jax.device_put(
            jax.tree_map(jnp.array, self.state), sharding.replicate()
        )

    def restore_checkpoint(self, path: str) -> "PARLIQLAgent":
        """Restore the policy from a checkpoint."""
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint = orbax_checkpointer.restore(path)
        self.state = JaxRLTrainState.create(
            apply_fn=self.state.apply_fn,
            params=checkpoint["state"]["params"],
            txs=self.state.txs,
            target_params=checkpoint["state"]["target_params"],
            rng=checkpoint["state"]["rng"],
        )

        return self

    def save_checkpoint(self, save_dir: str, step: int, overwrite: bool = True) -> None:
        """Save the policy to a checkpoint."""
        checkpoint = {
            "state": jax.device_get(self.state),
        }
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(checkpoint)
        save_path = tf.io.gfile.join(save_dir, f"checkpoint_{step}")
        os.makedirs(save_path, exist_ok=True)
        if overwrite and tf.io.gfile.exists(save_path):
            tf.io.gfile.rmtree(save_path)
        orbax_checkpointer.save(save_path, checkpoint, save_args=save_args)

    def replace(self, **kwargs):
        # IQL is a flax.struct.PyTreeNode, so it uses replace to update its state attribute during
        # the update function.
        return dict(kwargs)
