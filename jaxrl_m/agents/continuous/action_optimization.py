from collections import namedtuple
from functools import partial
from typing import Dict, Optional, Tuple, Union

import chex
import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import tensorflow as tf
from jaxrl_m.agents.continuous.base_policy import BasePolicy, BasePolicyTypes
from jaxrl_m.common.common import (
    JaxRLTrainState,
    ModuleDict,
    make_dict_kwargs_hashable_decorator,
)
from jaxrl_m.common.typing import Batch
from jaxrl_m.utils.timer_utils import Timer

LocalOptimizationState = namedtuple(
    "LocalOptimizationState",
    [
        "actions",
        "last_gradient_norm",
        "action_with_max_value",
        "max_value",
        "action_with_max_value_index",
    ],
)


@partial(
    jax.jit,
    static_argnames=(
        # Static critic agent to support non-hashable PyTrees
        "critic",
        "num_steps",
        "optimize_critic_ensemble_min",
        "use_target_critic",
        "keep_action_with_max_value",
    ),
)
def local_optimization_steps(
    observations: Union[
        jnp.ndarray, Dict[str, jnp.ndarray]
    ],  # dict will not be made hashable bc
    # it's not kwarg
    base_policy_actions: jnp.ndarray,
    critic: flax.struct.PyTreeNode,
    # non-static critic state though!
    critic_state: flax.struct.PyTreeNode,
    num_steps: int,
    step_size: float,
    optimize_critic_ensemble_min: bool = True,
    action_space_low: Optional[jnp.ndarray] = None,
    action_space_high: Optional[jnp.ndarray] = None,
    use_target_critic: bool = False,
    keep_action_with_max_value: bool = True,
) -> LocalOptimizationState:
    """Take num_steps steps maximizing the critic."""
    if type(observations) is dict:
        key = list(observations.keys())[0]
        batch_size = observations[key].shape[0]
    else:
        assert len(observations.shape) == 2
        batch_size = observations.shape[0]
    assert len(base_policy_actions.shape) == 2

    # action is first to differentiate wrt it
    def critic_fn(action, obs):
        critic_params = (
            critic_state.params if not use_target_critic else critic_state.target_params
        )
        critic_values = critic_state.apply_fn(
            {
                "params": critic_params,
            },
            obs,
            action,
            name="critic",
        )
        if type(critic_values) is tuple:
            # It's a distributional critic
            critic_values = critic_values[0]
        chex.assert_shape(
            critic_values, (critic.config["critic_ensemble_size"], action.shape[0])
        )
        if optimize_critic_ensemble_min:
            return critic_values.min(axis=0).sum(), critic_values.min(axis=0)
        return critic_values.mean(axis=0).sum(), critic_values.mean(axis=0)

    def get_critic_gradient_wrt_actions(obs: jnp.ndarray, action: jnp.ndarray):
        return jax.grad(critic_fn, has_aux=True)(action, obs)

    def body(i, state):
        actions = state.actions

        critic_gradient_wrt_actions, current_values = get_critic_gradient_wrt_actions(
            observations, actions
        )
        chex.assert_shape(current_values, (batch_size,))
        chex.assert_shape(
            critic_gradient_wrt_actions, (batch_size, base_policy_actions.shape[1])
        )
        # Update the action with the maximum value, each action in the batch independently
        if keep_action_with_max_value:
            max_value = jnp.maximum(state.max_value, current_values)
            action_with_max_value = jnp.where(
                jnp.broadcast_to(
                    current_values[..., None], state.action_with_max_value.shape
                )
                == max_value[..., None],
                state.actions,
                state.action_with_max_value,
            )
            action_with_max_value_index = jnp.where(
                current_values == max_value,
                i,
                state.action_with_max_value_index,
            )
            chex.assert_shape(action_with_max_value_index, (batch_size,))

        else:
            action_with_max_value = state.action_with_max_value
            max_value = state.max_value
            action_with_max_value_index = state.action_with_max_value_index

        actions = actions + step_size * critic_gradient_wrt_actions
        # Clip actions to the action space
        if action_space_low is not None and action_space_high is not None:
            chex.assert_shape(action_space_low, base_policy_actions.shape[-1:])
            chex.assert_shape(action_space_high, base_policy_actions.shape[-1:])
            actions = jnp.clip(actions, action_space_low, action_space_high)

        gradient_norm = jnp.linalg.norm(critic_gradient_wrt_actions, axis=-1).mean()
        return LocalOptimizationState(
            actions=actions,
            last_gradient_norm=gradient_norm,
            action_with_max_value=action_with_max_value,
            max_value=max_value,
            action_with_max_value_index=action_with_max_value_index,
        )

    initial_state = LocalOptimizationState(
        actions=base_policy_actions,
        last_gradient_norm=jnp.inf,
        action_with_max_value=base_policy_actions,
        max_value=jnp.broadcast_to(-jnp.inf, (batch_size,)),
        action_with_max_value_index=jnp.zeros(batch_size, dtype=jnp.int32),
    )

    final_state = jax.lax.fori_loop(
        0,
        num_steps,
        body,
        initial_state,
    )

    # Update the action with the maximum value one last time
    if keep_action_with_max_value:
        current_values = critic_fn(final_state.actions, observations)[1]
        max_value = jnp.maximum(final_state.max_value, current_values)
        action_with_max_value = jnp.where(
            jnp.broadcast_to(
                current_values[..., None], final_state.action_with_max_value.shape
            )
            == max_value[..., None],
            final_state.actions,
            final_state.action_with_max_value,
        )
        action_with_max_value_index = jnp.where(
            current_values == max_value,
            num_steps,
            final_state.action_with_max_value_index,
        )

    return final_state._replace(
        actions=(
            action_with_max_value if keep_action_with_max_value else final_state.actions
        ),
        action_with_max_value=action_with_max_value,
        action_with_max_value_index=action_with_max_value_index,
    )


def test_local_optimization_steps():
    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    observations = jax.random.normal(
        key,
        shape=(
            3,
            3,
        ),
    )  # (batch_size, obs_dim)
    actions = jax.random.normal(
        key,
        shape=(
            3,
            3,
        ),
    )  # (batch_size, act_dim)

    class TestCriticModule(nn.Module):
        @nn.compact
        def __call__(self, obs, act):
            assert len(obs.shape) == 2
            return (
                jnp.array(
                    [
                        2 * act[:, 0] * obs[:, 0]
                        - act[:, 1] * obs[:, 1]
                        + act[:, 2] * obs[:, 2]
                    ]
                )
                .reshape(1, obs.shape[0])
                .repeat(2, axis=0)
                .squeeze()
            )

    model_def = ModuleDict({"critic": TestCriticModule()})
    state = JaxRLTrainState.create(
        apply_fn=model_def.apply,
        params={},
        txs=None,
        target_params={},
        rng=rng,
    )

    class TestAgent(flax.struct.PyTreeNode):
        state: JaxRLTrainState
        config = {"critic_ensemble_size": 2}

    critic = TestAgent(state)
    num_steps = 10
    step_size = 10

    local_optimization_results = local_optimization_steps(
        observations, actions, critic, state, num_steps, step_size
    )
    expected_actions = (
        actions
        + num_steps
        * step_size
        * jnp.array([2, -1, 1]).reshape(1, -1).repeat(3, axis=0)
        * observations
    )
    assert jnp.all(jnp.isclose(local_optimization_results.actions, expected_actions)), (
        local_optimization_results.actions,
        expected_actions,
    )
    expected_last_gradient_norm = jnp.linalg.norm(
        jnp.array([2, -1, 1]).reshape(1, -1).repeat(3, axis=0) * observations,
        axis=1,
    ).mean()
    assert jnp.isclose(
        local_optimization_results.last_gradient_norm, expected_last_gradient_norm
    ), (
        local_optimization_results.last_gradient_norm,
        expected_last_gradient_norm,
    )
    assert local_optimization_results.num_gradient_steps_taken == num_steps


@make_dict_kwargs_hashable_decorator
@partial(
    jax.jit,
    static_argnames=(
        # Static critic agent to support non-hashable PyTrees
        "critic_agent",
        "num_base_policy_actions",
        "num_actions_to_keep",
        "num_steps",
        "optimize_critic_ensemble_min",
        "use_target_critic",
        "argmax",
    ),
)
def action_optimization_sample_actions(
    observations: jnp.ndarray,
    critic_agent: flax.struct.PyTreeNode,
    # non-static critic state though!
    critic_state: flax.struct.PyTreeNode,
    num_base_policy_actions: int,
    num_actions_to_keep: int,
    num_steps: int,
    step_size: float,
    optimize_critic_ensemble_min: bool,
    rng: jax.random.PRNGKey,
    action_space_low: jnp.ndarray,
    action_space_high: jnp.ndarray,
    use_target_critic: bool = False,
    argmax: bool = False,
    dataset_actions_to_consider: Optional[jnp.ndarray] = None,
) -> Tuple[distrax.Distribution, Dict[str, jnp.ndarray]]:
    info_metrics = {}
    if type(observations) is tuple and len(observations) == 2:
        # It's goal conditioned
        observations, goals = observations
    else:
        goals = None
    assert "base_policy_actions" in observations
    base_policy_actions = observations["base_policy_actions"]
    if len(base_policy_actions.shape) == 1:
        base_policy_actions = base_policy_actions[None]
    batch_size = base_policy_actions.shape[0]

    if "encoding" in observations:
        assert (
            False
        ), "encoding should not be calculated here, because critic and base policy use different encodings"
    elif "image" in observations:
        assert "proprio" in observations
        if len(observations["image"].shape) == 3:
            observations["image"] = observations["image"][None]
            observations["proprio"] = observations["proprio"][None]
            if goals is not None and len(goals["image"].shape) == 3:
                goals["image"] = goals["image"][None]
        proprio = observations["proprio"]
        if goals is not None:
            observations = (observations, goals)

        observations = {
            "encoding": critic_state.apply_fn(
                {"params": critic_state.params},
                observations,
                train=False,
                name="critic_encoder",
            ),
            "proprio": proprio,
        }

    else:
        assert "state" in observations
        if len(observations["state"].shape) == 1:
            observations["state"] = observations["state"][None]

    repeated_critic_observations = jax.tree_map(
        lambda x: x[:, None]
        .repeat(num_base_policy_actions, axis=1)
        .reshape(batch_size * num_base_policy_actions, *x.shape[1:]),
        observations,
    )

    base_policy_actions = base_policy_actions.reshape(
        batch_size * num_base_policy_actions, base_policy_actions.shape[-1]
    )

    # Add the base policy actions l1 norm to the metrics
    info_metrics["base_policy_actions_l1_norm"] = jnp.mean(
        jnp.sum(jnp.abs(base_policy_actions), axis=1)
    )

    rng, key = jax.random.split(rng)
    q_values_before = (
        critic_agent.forward_critic(
            (repeated_critic_observations),
            base_policy_actions,
            rng=key,
            grad_params=critic_state.params,
            train=False,
        )
        .mean(axis=0)
        .reshape(batch_size * num_base_policy_actions)
    )

    if num_actions_to_keep < num_base_policy_actions:
        # Keep only a subset of the actions, ranked by the critic

        q_values = q_values_before.reshape(batch_size, num_base_policy_actions)
        top_k_indices = jnp.argsort(q_values, axis=-1)[:, -num_actions_to_keep:]
        base_policy_actions = base_policy_actions.reshape(
            batch_size, num_base_policy_actions, base_policy_actions.shape[-1]
        )
        repeated_critic_observations = jax.tree_map(
            lambda x: x.reshape(
                batch_size,
                num_base_policy_actions,
                *x.shape[1:],
            ),
            repeated_critic_observations,
        )
        base_policy_actions = jnp.take_along_axis(
            base_policy_actions, top_k_indices[:, :, None], axis=1
        )
        # Best actions will be last in the list

        # top_k_indices needs to have the same number of dimensions as repeated_observations
        # for every key. Thus, for every key, we maintain the 2 existing dimensions, and add
        # None for every other dimension.
        repeated_critic_observations = jax.tree_map(
            lambda x: jnp.take_along_axis(
                x,
                top_k_indices[
                    (
                        slice(None),  # batch dimension
                        slice(None),  # num_actions_to_keep dimension
                        *(None,)
                        * (
                            len(x.shape) - len(top_k_indices.shape)
                        ),  # x's additional dimensions
                    )
                ],
                axis=1,
            ),
            repeated_critic_observations,
        )
        base_policy_actions = base_policy_actions.reshape(
            batch_size * num_actions_to_keep,
            base_policy_actions.shape[-1],
        )
        repeated_critic_observations = jax.tree_map(
            lambda x: x.reshape(
                batch_size * num_actions_to_keep,
                *x.shape[2:],
            ),
            repeated_critic_observations,
        )
        q_values_before = jnp.take_along_axis(
            q_values_before.reshape(batch_size, num_base_policy_actions),
            top_k_indices,
            axis=1,
        ).reshape(batch_size * num_actions_to_keep)

    chex.assert_shape(
        base_policy_actions,
        (
            batch_size * num_actions_to_keep,
            base_policy_actions.shape[-1],
        ),
    )
    info_metrics["base_policy_actions_q_values"] = q_values_before
    info_metrics["base_policy_actions_q_values_mean"] = q_values_before.mean()

    if dataset_actions_to_consider is not None:
        # add dataset actions to the ddpm actions
        chex.assert_shape(
            dataset_actions_to_consider, (batch_size, base_policy_actions.shape[-1])
        )
        base_policy_actions = jnp.concatenate(
            [
                base_policy_actions.reshape(
                    batch_size,
                    num_actions_to_keep,
                    base_policy_actions.shape[-1],
                )[
                    :, 1:, :
                ],  # skip the worst action
                dataset_actions_to_consider[:, None, :],
            ],
            axis=1,
        ).reshape(
            batch_size * (num_actions_to_keep),
            base_policy_actions.shape[-1],
        )

    # Propagate the actions through the Q function
    local_optimization_results: LocalOptimizationState = local_optimization_steps(
        repeated_critic_observations,
        base_policy_actions,
        critic=critic_agent,
        critic_state=critic_state,
        num_steps=num_steps,
        step_size=step_size,
        optimize_critic_ensemble_min=optimize_critic_ensemble_min,
        action_space_low=action_space_low,
        action_space_high=action_space_high,
        use_target_critic=use_target_critic,
    )
    actions = local_optimization_results.actions
    last_gradient_norm = local_optimization_results.last_gradient_norm

    assert actions.shape == base_policy_actions.shape, actions.shape

    # Add the last gradient norm and actions l1 norm to the metrics
    info_metrics["last_gradient_norm"] = last_gradient_norm
    info_metrics["local_optimization_actions_l1_norm"] = jnp.mean(
        jnp.sum(jnp.abs(actions), axis=1)
    )

    # Compute the Q values
    rng, key = jax.random.split(rng)
    q_values = critic_agent.forward_critic(
        repeated_critic_observations,
        actions,
        rng=key,
        grad_params=critic_state.params,
        train=False,
    ).mean(axis=0)
    info_metrics["q_values_after_steps"] = q_values.mean()
    info_metrics["total_value_increase"] = jnp.mean(q_values - q_values_before)
    info_metrics["l1(q_diff_actions, base_policy_actions)"] = jnp.mean(
        jnp.sum(jnp.abs(actions - base_policy_actions), axis=-1)
    )
    info_metrics["index of max value action"] = (
        local_optimization_results.action_with_max_value_index
    )
    logits = q_values.reshape(batch_size, num_actions_to_keep)

    if argmax:
        # Return Deterministic argmax action
        actions = actions.reshape(
            batch_size,
            num_actions_to_keep,
            actions.shape[-1],
        )
        action_indices = jnp.argmax(logits, axis=-1)
        actions = jnp.take_along_axis(
            actions, action_indices[..., None, None], axis=1
        ).reshape(batch_size, actions.shape[-1])
        return (
            distrax.Independent(
                distrax.Deterministic(loc=actions),
                reinterpreted_batch_ndims=1,
            ),
            info_metrics,
        )

    # Make a batch of Mixtures of deterministic values
    mixture_distribution = distrax.Categorical(logits=logits)
    info_metrics["categorical_entropy"] = mixture_distribution.entropy().mean()
    info_metrics["categorical_logits_min"] = jnp.mean(logits.min(axis=1))
    info_metrics["categorical_logits_max"] = jnp.mean(logits.max(axis=1))
    info_metrics["categorical_logits_mean"] = jnp.mean(logits)
    info_metrics["categorical_logits_std"] = jnp.mean(logits.std(axis=1))
    actions = actions.reshape(
        batch_size,
        num_actions_to_keep,
        actions.shape[-1],
    )

    components_distribution = distrax.Independent(
        distrax.Deterministic(actions), reinterpreted_batch_ndims=1
    )
    return (
        distrax.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            components_distribution=components_distribution,
        ),
        info_metrics,
    )


def add_base_policy_actions_to_batch(
    batch: Batch,
    base_policy_agent: BasePolicy,
    num_base_policy_actions: int,
    seed: jax.random.PRNGKey,
    save_dir: Optional[str] = None,
    epoch: Optional[int] = None,
    timer: Optional[Timer] = None,
    base_policy_type: Optional[BasePolicyTypes] = None,
    add_to_next_observations: bool = False,
    manual_cache_dir: Optional[str] = None,
) -> Batch:
    observation_keys = ["observations"]
    if add_to_next_observations:
        observation_keys.append("next_observations")
    cache_dir = manual_cache_dir
    if base_policy_type is not None and base_policy_type == BasePolicyTypes.OpenVLA:
        if manual_cache_dir is None:
            cache_dir = tf.io.gfile.join(
                save_dir,
                "base_policy_checkpoints_from_agent_trainer",
                f"checkpoint_{epoch-1}",
                "openvla_cache",
            )

    for observation_key in observation_keys:
        if not isinstance(batch[observation_key], dict):
            batch[observation_key] = {"state": batch[observation_key]}
        assert "base_policy_actions" not in batch[observation_key]
        batch[observation_key]["base_policy_actions"] = (
            base_policy_agent.sample_actions(
                # Add empty observation history axis
                jax.tree_map(lambda x: x[:, None], batch[observation_key]),
                repeat=num_base_policy_actions,
                cache_dir=cache_dir,
                timer=timer,
                seed=seed,
            )
        )

    return batch
