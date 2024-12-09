from typing import Optional, Union
import os
import gym
import gym.spaces
import numpy as np
import jax
import tensorflow as tf
from jaxrl_m.data.dataset import Dataset, DatasetDict


def _init_replay_dict(
    obs_space: gym.Space, capacity: int
) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError()


def _insert_recursively(
    dataset_dict: DatasetDict, data_dict: DatasetDict, insert_index: int
):
    if isinstance(dataset_dict, np.ndarray):
        dataset_dict[insert_index] = data_dict
    elif isinstance(dataset_dict, dict):
        assert dataset_dict.keys() == data_dict.keys()
        for k in dataset_dict.keys():
            _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
    else:
        raise TypeError()


class ReplayBuffer(Dataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        next_observation_space: Optional[gym.Space] = None,
        goal_space: Optional[gym.Space] = None,
        store_mc_return: bool = False,
        store_max_trajectory_reward: bool = False,
        seed: Optional[int] = None,
    ):
        if next_observation_space is None:
            next_observation_space = observation_space

        observation_data = _init_replay_dict(observation_space, capacity)
        next_observation_data = _init_replay_dict(next_observation_space, capacity)
        dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=np.empty((capacity, *action_space.shape), dtype=action_space.dtype),
            rewards=np.empty((capacity,), dtype=np.float32),
            masks=np.empty((capacity,), dtype=bool),
        )

        if goal_space is not None:
            goal_space = _init_replay_dict(goal_space, capacity)
            dataset_dict["goals"] = goal_space

        if store_mc_return:
            mc_returns = np.empty((capacity,), dtype=np.float32)
            dataset_dict["mc_returns"] = mc_returns

        if store_max_trajectory_reward:
            max_trajectory_rewards = np.empty((capacity,), dtype=np.float32)
            dataset_dict["max_trajectory_rewards"] = max_trajectory_rewards

        super().__init__(dataset_dict, seed)

        self._size = 0
        self._capacity = capacity
        self._insert_index = 0

    def __len__(self) -> int:
        return self._size

    def insert(self, data_dict: DatasetDict):
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def batched_insert(self, data_dict: DatasetDict):
        batch_size = data_dict["observations"].shape[0]
        insert_index = self._insert_index
        for i in range(batch_size):
            _insert_recursively(
                self.dataset_dict, {k: v[i] for k, v in data_dict.items()}, insert_index
            )
            insert_index = (insert_index + 1) % self._capacity

        self._insert_index = insert_index
        self._size = min(self._size + batch_size, self._capacity)

    def save(self, path: str):
        dict_to_save = jax.tree.map(lambda x: x[: self._size], self.dataset_dict)
        path = os.path.join(path, "replay_buffer.npz")
        if tf.io.gfile.exists(path):
            tf.io.gfile.remove(path)

        # with tf.io.gfile.GFile(path, "wb") as f:
        #     np.savez(f, **dict_to_save)

        # Save to "tmp.npz" first
        random_identifier = np.random.randint(0, 1e9)
        with open(f"./tmp_{random_identifier}.npz", "wb") as f:
            np.savez(f, **dict_to_save)

        # Move "tmp.npz" to the destination
        tf.io.gfile.copy(f"tmp_{random_identifier}.npz", path)
        tf.io.gfile.remove(f"tmp_{random_identifier}.npz")
