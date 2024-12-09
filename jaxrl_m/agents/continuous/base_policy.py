"""Interface that any PA-RL base policy should implement."""

from dataclasses import dataclass
from enum import Enum

import jax
from jaxrl_m.common.typing import Batch, Data
from jaxrl_m.utils.timer_utils import Timer


class BasePolicyTypes(Enum):
    OpenVLA = "openvla"
    DDPM = "ddpm"
    AutoRegressiveTransformer = "transformer"


class BasePolicy:
    """Prototype of PA-RL base policy."""

    def sample_actions(
        self,
        observations: Data,
        repeat: int = 1,
        cache_dir: str = None,
        timer: Timer = None,
    ):
        """Sample actions from the policy."""
        raise NotImplementedError

    def update(self, batch: Batch, timer: Timer) -> Data:
        """Perform one step of supervised learning on the batch."""
        raise NotImplementedError

    def prepare_for_finetuning(self):
        """Prepare the policy for finetuning."""
        pass

    def prepare_for_inference(self):
        """Prepare the policy for inference."""
        pass

    def restore_checkpoint(
        self, path: str, sharding: jax.sharding.Sharding
    ) -> "BasePolicy":
        """Restore the policy from a checkpoint."""
        raise NotImplementedError

    def save_checkpoint(
        self, save_dir: str, step: int, keep: int = 10, overwrite: bool = True
    ):
        """Save the policy to a checkpoint."""
        raise NotImplementedError

    def clear_cache(self):
        """Clear the actions cache."""
        pass

    def to_device(self, sharding: jax.sharding.Sharding):
        """Move policy params to device (gpu/tpu)."""
        pass
