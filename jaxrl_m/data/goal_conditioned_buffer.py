from typing import Iterable, List, Optional

import flax
import gym
import h5py  # NOQA
import jax
import numpy as np
import tensorflow as tf
from flax.core import FrozenDict, frozen_dict

from jaxrl_m.common.typing import Array, PRNGKey
from jaxrl_m.data.replay_buffer import ReplayBuffer
from jaxrl_m.data.tf_augmentations import augment, augment_batch


def _split_devices(data_dict, num_devices):
    if isinstance(data_dict, dict):
        batch = {}
        for k, v in data_dict.items():
            batch[k] = _split_devices(v, num_devices)
    elif isinstance(data_dict, Array):
        return np.array(np.split(data_dict, num_devices))
    else:
        raise TypeError("Unsupported type.")
    return batch


def _get_type_spec(data_dict):
    if isinstance(data_dict, dict):
        batch = {}
        for k, v in data_dict.items():
            batch[k] = _get_type_spec(v)
    elif isinstance(data_dict, Array):
        return tf.TensorSpec(data_dict.shape, dtype=data_dict.dtype)
    else:
        raise TypeError("Unsupported type.")
    return batch


class GCReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        goal_keys: List[str],
        capacity: int,
        fraction_next: int,
        fraction_uniform: int,
        fraction_last: int,
        fraction_negative: int,
        batch_size: int,
        reward_key: str = "image",
        reward_thresh: str = 0.3,
        seed: Optional[int] = None,
        action_metadata: Optional[dict] = None,
        num_devices: Optional[int] = 1,
        augment: bool = False,
        augment_next_obs_goal_differently: bool = False,
        augment_kwargs: Optional[dict] = None,
        prefetch_num_batches: int = 5,
    ):
        assert fraction_next + fraction_negative + fraction_uniform + fraction_last <= 1

        self.fraction_uniform = fraction_uniform
        self.fraction_next = fraction_next
        self.fraction_last = fraction_last
        self.fraction_negative = fraction_negative
        self.batch_size = batch_size
        self.reward_key = reward_key
        self.reward_thresh = reward_thresh
        self.relabel_configs = [
            # (mode, fraction, actor_loss_mask, reward)
            ("next", self.fraction_next, 1, 0),
            ("uniform", self.fraction_uniform, 1, -1),
            ("last", self.fraction_last, 1, -1),
            ("negative", self.fraction_negative, 0, -1),
        ]

        self.action_metadata = action_metadata
        self.num_devices = num_devices
        self.steps_remaining = np.empty((capacity,))
        self.augment = augment
        self.augment_kwargs = augment_kwargs
        self.augment_next_obs_goal_differently = augment_next_obs_goal_differently
        self.prefetch_num_batches = prefetch_num_batches
        self.rng = jax.random.PRNGKey(seed)

        # TODO: Support encoded goals.
        self.goal_keys = goal_keys
        goal_space = gym.spaces.Dict()
        for goal_key in self.goal_keys:
            goal_space[goal_key] = observation_space[goal_key]

        super().__init__(
            observation_space,
            action_space,
            goal_space=goal_space,
            capacity=capacity,
            seed=seed,
        )

    def insert(self, transition, steps_remaining):
        self.steps_remaining[self._insert_index] = steps_remaining
        super().insert(transition)

    def _sample(
        self, keys: Optional[Iterable[str]] = None, idx: Optional[np.ndarray] = None
    ) -> frozen_dict.FrozenDict:
        if idx is None:
            if hasattr(self.np_random, "integers"):
                idx = self.np_random.integers(len(self), size=self.batch_size)
            else:
                idx = self.np_random.randint(len(self), size=self.batch_size)
        batch = super().sample(self.batch_size, keys, idx)
        batch = batch.unfreeze()

        batch["actor_loss_mask"] = np.ones(self.batch_size)

        source_indices = {
            "uniform": self.np_random.integers(
                idx, idx + self.steps_remaining[idx] + 1
            ),
            "last": idx + self.steps_remaining[idx],
            "negative": self.np_random.integers(len(self), size=self.batch_size),
        }

        start = 0
        breakpoint()
        for mode, fraction, actor_loss_mask, reward in self.relabel_configs:
            if fraction == 0.0:
                continue

            num_entries = int(self.batch_size * fraction)
            end = start + num_entries

            if mode == "next":
                source = batch["next_observations"]
                source_idx = np.arange(start, end)
            elif mode in source_indices.keys():
                source = self.dataset_dict["next_observations"]
                source_idx = source_indices[mode][start:end]

            for key in self.goal_keys:
                batch["goals"][key][start:end] = source[key][source_idx]

            batch["actor_loss_mask"][start:end] = float(actor_loss_mask)
            # batch['rewards'][start:end] = float(reward)

        batch["rewards"] = self.compute_rewards(batch["observations"], batch["goals"])

        if self.action_metadata is not None:
            batch["actions"] = (
                batch["actions"] - self.action_metadata["mean"]
            ) / self.action_metadata["std"]
            batch["actions"] = batch["actions"].astype(np.float32)

        return batch

    def sample(
        self, keys: Optional[Iterable[str]] = None, idx: Optional[np.ndarray] = None
    ) -> frozen_dict.FrozenDict:
        batch = self._sample(keys, idx)
        if self.augment:
            self.rng, key = jax.random.split(self.rng)
            batch = self._augment(key, batch)

        batch = _split_devices(batch, self.num_devices)

        return batch

    def _augment(self, seed, image):
        if len(seed.shape) > 0:
            sub_seeds = [seed]
        else:
            sub_seeds = [[seed, seed]]
        if self.augment_next_obs_goal_differently:
            for _ in range(2):
                sub_seeds.append(
                    tf.random.stateless_uniform(
                        [2],
                        seed=sub_seeds[-1],
                        minval=None,
                        maxval=None,
                        dtype=tf.int32,
                    )
                )
        else:
            sub_seeds *= 3

        for key, sub_seed in zip(
            ["observations", "next_observations", "goals"], sub_seeds
        ):
            if len(image[key]["image"].shape) > 3:
                image[key]["image"] = augment_batch(
                    image[key]["image"], sub_seed, **self.augment_kwargs
                )
            else:
                image[key]["image"] = augment(
                    image[key]["image"], sub_seed, **self.augment_kwargs
                )
        return image

    def compute_rewards(self, observations, goals):
        s = observations[self.reward_key]
        g = goals[self.reward_key]

        s = np.reshape(s, [s.shape[0], -1])
        g = np.reshape(g, [g.shape[0], -1])

        dists = np.linalg.norm(s - g, axis=1)
        successes = dists <= self.reward_thresh
        rewards = successes.astype(np.float32) - 1.0

        return rewards

    def get_iterator(self):
        def gen():
            while True:
                yield self._sample()

        sample_batch = self._sample()
        type_spec = _get_type_spec(sample_batch)
        dataset = tf.data.Dataset.from_generator(gen, output_signature=type_spec)

        if self.augment:
            dataset = dataset.unbatch()
            dataset = dataset.enumerate(start=0)
            dataset = dataset.map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.batch(self.batch_size // self.num_devices)
        else:
            dataset = dataset.rebatch(self.batch_size // self.num_devices)
        dataset = dataset.batch(self.num_devices)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        iterator = dataset.as_numpy_iterator()
        return flax.jax_utils.prefetch_to_device(
            iterator,
            size=2,
            devices=jax.local_devices()[: self.num_devices],
        )


def load_gc_trajectories(trajectories, replay_buffer):
    for trajectory in trajectories:
        for i in range(len(trajectory["action"])):
            transition = {
                "goals": trajectory["goal"][i],
                "observations": trajectory["observation"][i],
                "next_observations": trajectory["next_observation"][i],
                "actions": trajectory["action"][i],
                "rewards": 0.0,
                "masks": 1.0,
            }
            replay_buffer.insert(transition, trajectory["steps_remaining"][i])


def load_gc_roboverse_data(paths, replay_buffer, use_proportion):
    for path in paths:
        with tf.io.gfile.GFile(path) as p:
            f = h5py.File(p, "r")
            use_trans_number = int(use_proportion * len(f["actions"]))
            # round to nearest trajectory boundary
            use_trans_number += f["steps_remaining"][use_trans_number - 1]
            data = {
                "observations/image": f["observations/images0"][:use_trans_number],
                "next_observations/image": f["next_observations/images0"][
                    :use_trans_number
                ],
                "observations/proprio": f["observations/state"][
                    :use_trans_number
                ].astype(np.float32),
                "next_observations/proprio": f["next_observations/state"][
                    :use_trans_number
                ].astype(np.float32),
                "actions": f["actions"][:use_trans_number],
                "masks": np.logical_not(f["terminals"][:use_trans_number]),
                "steps_remaining": f["steps_remaining"][:use_trans_number],
            }
            for i in range(use_trans_number):
                transition = {
                    "observations": {
                        "image": data["observations/image"][i],
                        "proprio": data["observations/proprio"][i],
                    },
                    "next_observations": {
                        "image": data["next_observations/image"][i],
                        "proprio": data["next_observations/proprio"][i],
                    },
                    "actions": data["actions"][i],
                    "rewards": 0.0,
                    "masks": data["masks"][i],
                    "goals": {
                        "image": data["next_observations/image"][
                            i + data["steps_remaining"][i]
                        ],
                    },
                }
                replay_buffer.insert(transition, data["steps_remaining"][i])
