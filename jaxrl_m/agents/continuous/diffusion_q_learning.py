from typing import Optional, Tuple, Union
import copy
from functools import partial
from overrides import overrides
from ml_collections import ConfigDict
import chex
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import distrax
from jaxrl_m.common.typing import Batch, Data, Params, PRNGKey
from jaxrl_m.common.common import JaxRLTrainState, ModuleDict
from jaxrl_m.common.optimizers import make_optimizer
from jaxrl_m.networks.mlp import MLP, LayerInputMLP, MLPResNet
from jaxrl_m.networks.actor_critic_nets import (
    Critic,
    DistributionalCritic,
    Policy,
    ensemblize,
)
from jaxrl_m.networks.distributional import hl_gauss_transform
from jaxrl_m.networks.lagrange import GeqLagrangeMultiplier, LeqLagrangeMultiplier
from jaxrl_m.agents.continuous.sac import SACAgent
from jaxrl_m.agents.continuous.ddpm_bc import ddpm_bc_loss
from jaxrl_m.networks.diffusion_nets import (
    FourierFeatures,
    ScoreActor,
    cosine_beta_schedule,
    vp_beta_schedule,
)


class DiffusionQLearningAgent(SACAgent):
    @overrides
    def _compute_next_actions(self, batch, rng, **kwargs):
        """
        compute the next actions but with repeat cql_n_actions times
        this should only be used when calculating critic loss using
        cql_max_target_backup
        """
        if not self.config["cql_max_target_backup"]:
            return super()._compute_next_actions(batch, rng, **kwargs)

        original_batch_size = batch["actions"].shape[0]

        # Repeat the batch cql_n_actions times
        batch = jax.tree.map(
            lambda x: jnp.repeat(x, self.config["cql_n_actions"], axis=0), batch
        )

        # Compute the next actions
        next_actions, next_actions_log_probs = super()._compute_next_actions(
            batch, rng, **kwargs
        )

        next_actions = jnp.reshape(
            next_actions, (original_batch_size, self.config["cql_n_actions"], -1)
        )

        next_actions_log_probs = jnp.reshape(
            next_actions_log_probs,
            (original_batch_size, self.config["cql_n_actions"]),
        )

        return next_actions, next_actions_log_probs

    @overrides
    def _process_target_next_qs(self, target_next_qs, next_actions_log_probs):
        """add cql_max_target_backup option"""

        if self.config["cql_max_target_backup"]:
            assert len(target_next_qs.shape) == 2
            max_target_indices = jnp.expand_dims(
                jnp.argmax(target_next_qs, axis=-1), axis=-1
            )
            target_next_qs = jnp.take_along_axis(
                target_next_qs, max_target_indices, axis=-1
            ).squeeze(-1)
            next_actions_log_probs = jnp.take_along_axis(
                next_actions_log_probs, max_target_indices, axis=-1
            ).squeeze(-1)

        target_next_qs = super()._process_target_next_qs(
            target_next_qs,
            next_actions_log_probs,
        )

        return target_next_qs

    @overrides
    def forward_policy(
        self,
        observations: Union[Data, Tuple[Data, Data]],
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
        temperature: float = 1.0,
        **kwargs,
    ) -> distrax.Distribution:
        def fn(input_tuple, time):
            current_x, rng = input_tuple
            input_time = jnp.broadcast_to(time, (current_x.shape[0], 1))

            eps_pred = self.state.apply_fn(
                {"params": grad_params or self.state.target_params},
                observations,
                current_x,
                input_time,
                name="actor",
            )

            alpha_1 = 1 / jnp.sqrt(self.config["alphas"][time])
            alpha_2 = (1 - self.config["alphas"][time]) / (
                jnp.sqrt(1 - self.config["alpha_hat"][time])
            )
            current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

            rng, key = jax.random.split(rng)
            z = jax.random.normal(
                key,
                shape=current_x.shape,
            )
            z_scaled = temperature * z
            current_x = current_x + (time > 0) * (
                jnp.sqrt(self.config["betas"][time]) * z_scaled
            )

            if self.config["clip_sampler"]:
                current_x = jnp.clip(
                    current_x, self.config["action_min"], self.config["action_max"]
                )

            return (current_x, rng), ()

        rng, key = jax.random.split(rng)

        batch_size = observations.shape[0]

        input_tuple, () = jax.lax.scan(
            fn,
            (jax.random.normal(key, (batch_size, *self.config["action_dim"])), rng),
            jnp.arange(self.config["diffusion_steps"] - 1, -1, -1),
        )

        for _ in range(self.config["repeat_last_step"]):
            input_tuple, () = fn(input_tuple, 0)

        action_0, rng = input_tuple
        return distrax.Independent(
            distrax.Deterministic(action_0), reinterpreted_batch_ndims=1
        )

    @overrides
    def policy_loss_fn(self, batch, params: Params, rng: PRNGKey):
        rl_loss, rl_loss_info = super().policy_loss_fn(batch, params, rng)

        # Calculate DDPM loss
        key, rng = jax.random.split(rng)
        time = jax.random.randint(
            key, (batch["actions"].shape[0],), 0, self.config["diffusion_steps"]
        )
        key, rng = jax.random.split(rng)
        noise_sample = jax.random.normal(key, batch["actions"].shape)

        alpha_hats = self.config["alpha_hat"][time]
        time = time[:, None]
        alpha_1 = jnp.sqrt(alpha_hats)[:, None]
        alpha_2 = jnp.sqrt(1 - alpha_hats)[:, None]

        noisy_actions = alpha_1 * batch["actions"] + alpha_2 * noise_sample

        rng, key = jax.random.split(rng)
        noise_pred = self.state.apply_fn(
            {"params": params},  # gradient flows through here
            batch["observations"],
            noisy_actions,
            time,
            train=True,
            rngs={"dropout": key},
            name="actor",
        )

        ddpm_loss, ddpm_loss_info = ddpm_bc_loss(
            noise_pred,
            noise_sample,
        )
        assert ddpm_loss.ndim == 0 and rl_loss.ndim == 0

        q_loss = rl_loss / jax.lax.stop_gradient(jnp.abs(rl_loss).mean())
        loss = ddpm_loss + q_loss * self.config["rl_weight"]

        info = {
            **rl_loss_info,
            **ddpm_loss_info,
            "q_loss": q_loss,
        }
        return loss, info

    def update(self, *args, **kwargs) -> Data:
        """
        Take one gradient step on all (or a subset) of the networks in the agent.

        Parameters:
            batch: Batch of data to use for the update. Should have keys:
                "observations", "actions", "next_observations", "rewards", "masks".
            pmap_axis: Axis to use for pmap (if None, no pmap is used).
            networks_to_update: Names of networks to update (default: all networks).
                For example, in high-UTD settings it's common to update the critic
                many times and only update the actor (and other networks) once.
            specified_neg_goals: if not None, use these negative goals for the
                goal-conditioned agent instead of sampling in the batch
        Returns:
            info dict.
        """
        new_state, info = self._update(*args, **kwargs)
        self.state = new_state
        return info

    @partial(jax.jit, static_argnames=("pmap_axis", "networks_to_update"))
    def _update(
        self,
        batch: Batch,
        *,
        pmap_axis: str = None,
        networks_to_update: frozenset[str] = frozenset(
            {"actor", "critic", "temperature"}
        ),
        specified_neg_goals: Optional[Data] = None,
    ) -> Tuple["SACAgent", dict]:

        batch_size = batch["rewards"].shape[0]
        chex.assert_tree_shape_prefix(batch, (batch_size,))

        rng, goal_rng = jax.random.split(self.state.rng)
        if self.config["goal_conditioned"]:
            new_goals, new_rewards, new_mc_returns = self._sample_negative_goals(
                batch, goal_rng
            )

        # Compute gradients and update params
        loss_fns = self.loss_fns(batch)

        # Only compute gradients for specified steps
        assert networks_to_update.issubset(
            loss_fns.keys()
        ), f"Invalid gradient steps: {networks_to_update}"
        for key in loss_fns.keys() - networks_to_update:
            loss_fns[key] = lambda params, rng: (0.0, {})

        new_state, info = self.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )

        # Update target network (if requested)
        if "critic" in networks_to_update:
            new_state = new_state.target_update(self.config["soft_target_update_rate"])

        # Update RNG
        new_state = new_state.replace(rng=rng)

        # Log learning rates
        for name, opt_state in new_state.opt_states.items():
            if (
                hasattr(opt_state, "hyperparams")
                and "learning_rate" in opt_state.hyperparams.keys()
            ):
                info[f"{name}_lr"] = opt_state.hyperparams["learning_rate"]

        return new_state, info

    def __init__(
        self,
        rng: jax.random.PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Model architecture
        encoder_def: nn.Module,
        shared_encoder: bool = False,
        distributional_critic: bool = False,
        distributional_critic_kwargs: dict = {
            "q_min": -100.0,
            "q_max": 0.0,
            "num_bins": 128,
        },
        critic_network_type: str = "mlp",
        critic_network_kwargs: dict = {
            "hidden_dims": [256, 256],
            "activate_final": True,
            "use_layer_norm": False,
        },
        critic_kwargs: dict = {},
        score_network_kwargs: dict = {
            "time_dim": 16,
            "num_blocks": 3,
            "dropout_rate": 0.0,
            "hidden_dim": 256,
            "use_layer_norm": False,
        },
        # DDPM algorithm train + inference config
        beta_schedule: str = "vp",
        diffusion_steps: int = 5,
        action_samples: int = 64,
        repeat_last_step: int = 0,
        target_update_rate=0.002,
        dropout_target_networks=True,
        **kwargs,
    ):
        # update algorithm config
        config = get_default_config(updates=kwargs)

        create_encoder_fn = partial(
            super()._create_encoder_def,
            use_proprio=False,
            proprioceptive_dims=None,
            enable_stacking=False,
            goal_conditioned=config.goal_conditioned,
            early_goal_concat=False,
            shared_goal_encoder=True,
        )

        if shared_encoder:
            encoder_def = create_encoder_fn(
                encoder_def=encoder_def, stop_gradient=False
            )
            encoders = {
                "actor": encoder_def,
                "critic": encoder_def,
            }
        else:
            encoders = {
                "actor": create_encoder_fn(
                    encoder_def=encoder_def, stop_gradient=False
                ),
                "critic": create_encoder_fn(
                    encoder_def=copy.deepcopy(encoder_def), stop_gradient=False
                ),
            }

        # Define networks
        policy_def = ScoreActor(
            encoder_def,
            FourierFeatures(score_network_kwargs["time_dim"], learnable=True),
            MLP(
                (
                    2 * score_network_kwargs["time_dim"],
                    score_network_kwargs["time_dim"],
                )
            ),
            MLPResNet(
                score_network_kwargs["num_blocks"],
                # actions.shape[-2] * actions.shape[-1],
                actions.shape[-1],
                dropout_rate=score_network_kwargs["dropout_rate"],
                use_layer_norm=score_network_kwargs["use_layer_norm"],
            ),
            name="actor",
        )

        if critic_network_type == "mlp":
            critic_network_class = MLP
        elif critic_network_type == "layer_input_mlp":  # PTR style
            critic_network_class = LayerInputMLP
        else:
            raise NotImplementedError(
                f"critic_network_type={critic_network_type} not implemented"
            )

        critic_backbone = partial(critic_network_class, **critic_network_kwargs)
        critic_backbone = ensemblize(critic_backbone, config.critic_ensemble_size)(
            name="critic_ensemble"
        )
        if distributional_critic:
            q_min = distributional_critic_kwargs["q_min"]
            q_max = distributional_critic_kwargs["q_max"]
            critic_def = partial(
                DistributionalCritic,
                encoder=encoders["critic"],
                network=critic_backbone,
                q_low=q_min,
                q_high=q_max,
                num_bins=distributional_critic_kwargs["num_bins"],
                **critic_kwargs,
            )(name="critic")
            scalar_target_to_dist_fn = hl_gauss_transform(
                min_value=q_min,
                max_value=q_max,
                num_bins=distributional_critic_kwargs["num_bins"],
            )[0]
        else:
            critic_def = partial(
                Critic,
                encoder=encoders["critic"],
                network=critic_backbone,
                **critic_kwargs,
            )(name="critic")
            scalar_target_to_dist_fn = None

        config["distributional_critic"] = distributional_critic
        config["scalar_target_to_dist_fn"] = scalar_target_to_dist_fn

        temperature_def = GeqLagrangeMultiplier(
            init_value=config.temperature_init,
            constraint_shape=(),
            name="temperature",
        )

        # model def
        networks = {
            "actor": policy_def,
            "critic": critic_def,
            "temperature": temperature_def,
            "actor_encoder": encoders["actor"],
            "critic_encoder": encoders["critic"],
        }

        model_def = ModuleDict(networks)

        # Define optimizers
        txs = {
            "actor": make_optimizer(**config.actor_optimizer_kwargs),
            "critic": make_optimizer(**config.critic_optimizer_kwargs),
            "temperature": make_optimizer(**config.temperature_optimizer_kwargs),
        }

        # init params
        rng, init_rng = jax.random.split(rng)
        extra_kwargs = {}

        assert len(actions.shape) == 2
        example_time = jnp.zeros((actions.shape[0], 1))

        params = model_def.init(
            init_rng,
            actor=[observations, actions, example_time],
            critic=[observations, actions],
            temperature=[],
            actor_encoder=[observations, {"train": False}],
            critic_encoder=[observations, {"train": False}],
            **extra_kwargs,
        )["params"]

        # create
        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )

        if beta_schedule == "cosine":
            betas = jnp.array(cosine_beta_schedule(diffusion_steps))
        elif beta_schedule == "linear":
            betas = jnp.linspace(1e-4, 2e-2, diffusion_steps)
        elif beta_schedule == "vp":
            betas = jnp.array(vp_beta_schedule(diffusion_steps))
        else:
            raise NotImplementedError(f"beta_schedule={beta_schedule} not implemented")

        alphas = 1 - betas
        alpha_hat = jnp.array(
            [jnp.prod(alphas[: i + 1]) for i in range(diffusion_steps)]
        )

        # config
        if config.target_entropy >= 0.0:
            config.target_entropy = -actions.shape[-1]
        config.betas = betas
        config.alphas = alphas
        config.alpha_hat = alpha_hat
        config.diffusion_steps = diffusion_steps
        config.action_samples = action_samples
        config.repeat_last_step = repeat_last_step
        config.target_update_rate = target_update_rate
        config.dropout_target_networks = dropout_target_networks
        config.action_dim = (actions.shape[-1],)
        config.policy_std_parameterization = "fixed"
        config = flax.core.FrozenDict(config)

        self.state = state
        self.config = config


def get_default_config(updates=None):
    config = ConfigDict()
    config.discount = 0.99
    config.backup_entropy = False
    config.target_entropy = 0.0
    config.soft_target_update_rate = 5e-3
    config.distributional_critic = False
    config.critic_ensemble_size = 2
    config.critic_subsample_size = None
    config.autotune_entropy = True
    config.temperature_init = 1.0
    config.actor_optimizer_kwargs = ConfigDict(
        {
            "learning_rate": 3e-4,
            "warmup_steps": 0,
        }
    )
    config.critic_optimizer_kwargs = ConfigDict(
        {
            "learning_rate": 3e-4,
            "warmup_steps": 0,
        }
    )
    config.temperature_optimizer_kwargs = ConfigDict(
        {
            "learning_rate": 3e-4,
        }
    )

    # Goal-conditioning
    config.goal_conditioned = False
    config.gc_kwargs = ConfigDict(
        {
            "negative_proportion": 0.0,
        }
    )

    config.early_goal_concat = False

    # DDPM
    config.clip_sampler = True
    config.action_min = -1.0
    config.action_max = 1.0
    config.image_observations = False

    config.bc_regularization_weight = 0.0
    config.rl_weight = 1.0
    config.cql_max_target_backup = False
    config.cql_n_actions = 10

    if updates is not None:
        config.update(ConfigDict(updates).copy_and_resolve_references())
    return config
