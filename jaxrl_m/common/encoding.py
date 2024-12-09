from typing import Dict, Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange, repeat


class EncodingWrapper(nn.Module):
    """
    Encodes observations into a single flat encoding, adding additional
    functionality for adding proprioception and stopping the gradient.

    Args:
        encoder: The encoder network.
        use_proprio: Whether to concatenate proprioception (after encoding).
        stop_gradient: Whether to stop the gradient after the encoder.
    """

    encoder: nn.Module
    use_proprio: bool
    stop_gradient: bool
    proprioceptive_dims: Optional[int] = None
    enable_stacking: bool = False

    def __call__(
        self, observations: Dict[str, jnp.ndarray], train: bool
    ) -> jnp.ndarray:
        # import pdb; pdb.set_trace()
        if (
            isinstance(observations, flax.core.FrozenDict)
            or isinstance(observations, dict)
            and ("image" in observations or "encoding" in observations)
        ):
            if "encoding" in observations:
                return observations["encoding"]
            obs = observations["image"]
            if self.enable_stacking:
                # Combine stacking and channels into a single dimension
                if len(obs.shape) == 4:
                    obs = rearrange(obs, "T H W C -> H W (T C)")
                if len(obs.shape) == 5:
                    obs = rearrange(obs, "B T H W C -> B H W (T C)")

        else:
            obs = observations

        encoding = self.encoder(obs, train=train)

        if self.use_proprio:
            proprio = observations["proprio"]
            if self.proprioceptive_dims is not None:
                # proprio = proprio[..., -self.proprioceptive_dims :]
                proprio = proprio[..., : self.proprioceptive_dims :]
            encoding = jnp.concatenate([encoding, proprio], axis=-1)
        if self.stop_gradient:
            encoding = jax.lax.stop_gradient(encoding)
        return encoding


class GCEncodingWrapper(nn.Module):
    """
    Encodes observations and goals into a single flat encoding. Handles all the
    logic about when/how to combine observations and goals.

    Takes a tuple (observations, goals) as input.

    Args:
        encoder: The encoder network for observations.
        goal_encoder: The encoder to use for goals (optional). If None, early
            goal concatenation is used, i.e. the goal is concatenated to the
            observation channel-wise before passing it through the encoder.
        use_proprio: Whether to concatenate proprioception (after encoding).
        stop_gradient: Whether to stop the gradient after the encoder.
    """

    encoder: nn.Module
    goal_encoder: Optional[nn.Module]
    use_proprio: bool
    stop_gradient: bool

    def __call__(
        self,
        observations_and_goals: Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]],
        train: bool,
    ) -> jnp.ndarray:
        if (
            isinstance(observations_and_goals, dict)
            and "encoding" in observations_and_goals
        ):
            return observations_and_goals["encoding"]

        observations, goals = observations_and_goals

        if "encoding" in observations:
            return observations["encoding"]

        if "image" not in observations and "image" not in goals:
            if len(observations.shape) == 3:
                obs_horizon = observations.shape[1]
                goals = repeat(goals, "B E -> B repeat E", repeat=obs_horizon)
            return self.encoder(
                jnp.concatenate([observations, goals], axis=-1), train=train
            )

        if len(observations["image"].shape) == 5:
            # obs history case
            batch_size, obs_horizon = observations["image"].shape[:2]
            # fold batch_size into obs_horizon to encode each frame separately
            obs_image = rearrange(observations["image"], "B T H W C -> (B T) H W C")
            # repeat goals so that there's a goal for each frame
            goal_image = repeat(
                goals["image"], "B H W C -> (B repeat) H W C", repeat=obs_horizon
            )
        else:
            obs_image = observations["image"]
            goal_image = goals["image"]

        if self.goal_encoder is None:
            # early goal concat
            encoder_inputs = jnp.concatenate([obs_image, goal_image], axis=-1)
            encoding = self.encoder(encoder_inputs, train=train)
        else:
            # late fusion
            encoding = self.encoder(obs_image, train=train)
            goal_encoding = self.goal_encoder(goals["image"], train=train)
            encoding = jnp.concatenate([encoding, goal_encoding], axis=-1)

        if len(observations["image"].shape) == 5:
            # unfold obs_horizon from batch_size
            encoding = rearrange(
                encoding, "(B T) F -> B (T F)", B=batch_size, T=obs_horizon
            )

        if self.use_proprio:
            encoding = jnp.concatenate([encoding, observations["proprio"]], axis=-1)

        if self.stop_gradient:
            encoding = jax.lax.stop_gradient(encoding)

        return encoding


class LCEncodingWrapper(nn.Module):
    """
    Encodes observations and language instructions into a single flat encoding.

    Takes a tuple (observations, goals) as input, where goals contains the language instruction.

    Args:
        encoder: The encoder network for observations.
        use_proprio: Whether to concatenate proprioception (after encoding).
        stop_gradient: Whether to stop the gradient after the encoder.
    """

    encoder: nn.Module
    use_proprio: bool
    stop_gradient: bool

    def __call__(
        self,
        observations_and_goals: Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]],
    ) -> jnp.ndarray:
        observations, goals = observations_and_goals

        if len(observations["image"].shape) == 5:
            # obs history case
            batch_size, obs_horizon = observations["image"].shape[:2]
            # fold batch_size into obs_horizon to encode each frame separately
            obs_image = rearrange(observations["image"], "B T H W C -> (B T) H W C")
            # repeat language so that there's an instruction for each frame
            language = repeat(
                goals["language"], "B E -> (B repeat) E", repeat=obs_horizon
            )
        else:
            obs_image = observations["image"]
            language = goals["language"]

        encoding = self.encoder(obs_image, cond_var=language)

        if len(observations["image"].shape) == 5:
            # unfold obs_horizon from batch_size
            encoding = rearrange(
                encoding, "(B T) F -> B (T F)", B=batch_size, T=obs_horizon
            )

        if self.use_proprio:
            encoding = jnp.concatenate([encoding, observations["proprio"]], axis=-1)

        if self.stop_gradient:
            encoding = jax.lax.stop_gradient(encoding)

        return encoding
