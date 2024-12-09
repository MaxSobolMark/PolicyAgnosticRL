from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Union

import os
import numpy as np
import tensorflow as tf
from absl import logging

from jaxrl_m.data.tf_augmentations import augment as augment_fn
from jaxrl_m.data.tf_goal_relabeling import GOAL_RELABELING_FUNCTIONS


class ImageReplayBuffer:
    def __init__(
        self,
        data_paths: List[str],
        seed: int,
        goal_relabeling_strategy: Optional[str] = None,
        goal_relabeling_kwargs: dict = {},
        shuffle_buffer_size: int = 10000,
        cache: bool = False,
        train: bool = True,
        tfrecords_include_next_observations: bool = True,
        states_only: bool = False,
        augment: bool = False,
        augment_kwargs: dict = {},
        include_next_actions: bool = False,
    ):
        self.goal_relabeling_strategy = goal_relabeling_strategy
        self.goal_relabeling_kwargs = goal_relabeling_kwargs
        self.is_train = train
        self.cache = cache
        self.tfrecords_include_next_observations = tfrecords_include_next_observations
        self.states_only = states_only
        self.augment = augment
        self.augment_kwargs = augment_kwargs
        self.include_next_actions = include_next_actions

        dataset = self._construct_tf_dataset(data_paths, seed)

        if train:
            dataset = dataset.shuffle(shuffle_buffer_size, seed=seed).repeat()

            # if augment:
            #     dataset = dataset.enumerate(start=seed)
            #     dataset = dataset.map(
            #         self._augment, num_parallel_calls=tf.data.AUTOTUNE
            #     )

        # dataset = dataset.batch(
        #     batch_size,
        #     num_parallel_calls=tf.data.experimental.AUTOTUNE,
        #     drop_remainder=True,
        #     deterministic=not train,
        # )

        self.tf_dataset = dataset

    def _construct_tf_dataset(
        self, data_paths: List[str], seed: int
    ) -> tf.data.Dataset:
        # shuffle again using the dataset API so the files are read in a
        # different order every epoch
        dataset = tf.data.Dataset.from_tensor_slices(data_paths).shuffle(
            len(data_paths), seed
        )

        # yields raw serialized examples
        dataset = tf.data.TFRecordDataset(dataset, num_parallel_reads=tf.data.AUTOTUNE)

        # yields trajectories
        dataset = dataset.map(self._decode_example, num_parallel_calls=tf.data.AUTOTUNE)

        # cache before add_goals because add_goals introduces randomness
        if self.cache:
            dataset = dataset.cache()

        dataset = dataset.map(self._add_goals, num_parallel_calls=tf.data.AUTOTUNE)

        if self.states_only:
            dataset = dataset.map(
                lambda x: {
                    k: (
                        x[k]
                        if k not in ["observations", "next_observations", "goals"]
                        else x[k]["proprio"]
                    )
                    for k in x
                },
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        if self.augment:
            dataset = dataset.enumerate(start=seed)
            dataset = dataset.map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)

        # unbatch to yield individual transitions
        dataset = dataset.unbatch()

        return dataset

    # the expected type spec for the serialized examples
    PROTO_TYPE_SPEC = {
        "observations/images0": tf.uint8,
        "observations/state": tf.float32,
        # "next_observations/images0": tf.uint8,
        # "next_observations/state": tf.float32,
        "actions": tf.float32,
        # "terminals": tf.bool,
        # "truncates": tf.bool,
    }

    def _decode_example(self, example_proto):
        # decode the example proto according to PROTO_TYPE_SPEC
        if self.goal_relabeling_strategy is None:
            self.PROTO_TYPE_SPEC["rewards"] = tf.float32
            self.PROTO_TYPE_SPEC["masks"] = tf.float32
            self.PROTO_TYPE_SPEC["mc_returns"] = tf.float32
        if self.tfrecords_include_next_observations:
            self.PROTO_TYPE_SPEC["next_observations/images0"] = tf.uint8
            self.PROTO_TYPE_SPEC["next_observations/state"] = tf.float32
        if self.states_only:
            self.PROTO_TYPE_SPEC.pop("observations/images0")
            if self.tfrecords_include_next_observations:
                self.PROTO_TYPE_SPEC.pop("next_observations/images0")
        features = {
            key: tf.io.FixedLenFeature([], tf.string)
            for key in self.PROTO_TYPE_SPEC.keys()
        }
        parsed_features = tf.io.parse_single_example(example_proto, features)
        parsed_tensors = {
            key: tf.io.parse_tensor(parsed_features[key], dtype)
            for key, dtype in self.PROTO_TYPE_SPEC.items()
        }
        if not self.tfrecords_include_next_observations:
            states = parsed_tensors["observations/state"]
            parsed_tensors["observations/state"] = states[:-1]
            parsed_tensors["next_observations/state"] = states[1:]
            # if self.states_only:
            #     parsed_tensors["observations/images0"] = None
            #     parsed_tensors["next_observations/images0"] = None
            # else:
            if not self.states_only:
                images = parsed_tensors["observations/images0"]
                parsed_tensors["observations/images0"] = images[:-1]
                parsed_tensors["next_observations/images0"] = images[1:]

        if self.include_next_actions:
            # add the next action as part of the observation
            actions = parsed_tensors["actions"]
            parsed_tensors["actions"] = actions[:-1]
            parsed_tensors["next_actions"] = actions[1:]
            # Since we don't have the last next action, we need to remove the last observation
            for key in parsed_tensors:
                if "actions" not in key:
                    parsed_tensors[key] = parsed_tensors[key][:-1]
        # restructure the dictionary into the downstream format
        return {
            "observations": {
                "proprio": parsed_tensors["observations/state"],
                **(
                    {
                        "image": parsed_tensors["observations/images0"],
                    }
                    if not self.states_only
                    else {}
                ),
            },
            "next_observations": {
                "proprio": parsed_tensors["next_observations/state"],
                **(
                    {
                        "image": parsed_tensors["next_observations/images0"],
                    }
                    if not self.states_only
                    else {}
                ),
            },
            "actions": parsed_tensors["actions"],
            **(
                {
                    "next_actions": parsed_tensors["next_actions"],
                }
                if self.include_next_actions
                else {}
            ),
            "terminals": tf.zeros(
                tf.shape(parsed_tensors["actions"])[0], dtype=tf.bool
            ),
            "truncates": tf.zeros(
                tf.shape(parsed_tensors["actions"])[0], dtype=tf.bool
            ),
            **(
                {
                    "rewards": parsed_tensors["rewards"],
                    "masks": parsed_tensors["masks"],
                    "mc_returns": parsed_tensors["mc_returns"],
                }
                if "rewards" in parsed_tensors
                else {}
            ),
        }

    def _add_goals(self, traj):
        if self.goal_relabeling_strategy is not None:
            traj = GOAL_RELABELING_FUNCTIONS[self.goal_relabeling_strategy](
                traj, **self.goal_relabeling_kwargs
            )

        return traj

    def _augment(self, seed, image):
        keys_to_augment = ["observations", "next_observations"]
        if "goals" in image:
            keys_to_augment.append("goals")
        for key in keys_to_augment:
            image[key]["image"] = augment_fn(
                image[key]["image"], [seed, seed], **self.augment_kwargs
            )
        return image

    def iterator(self, batch_size):
        return (
            self.tf_dataset.batch(
                batch_size,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                drop_remainder=True,
                deterministic=not self.is_train,
            )
            .prefetch(tf.data.AUTOTUNE)
            .as_numpy_iterator()
        )


def save_trajectory_as_tfrecord(trajectory: Dict[str, np.ndarray], path: str):
    def tensor_feature(value):
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()])
        )

    assert path.endswith(".tfrecord")
    assert trajectory.keys() >= {
        "observations",
        "next_observations",
        "actions",
    }
    assert type(trajectory["observations"]) == list
    assert trajectory["observations"][0].keys() >= {"image", "proprio"}

    if tf.io.gfile.exists(path):
        print(f"Warning: Removing existing file at {path}")
        try:
            tf.io.gfile.remove(path)
        except Exception as e:
            breakpoint()

    tf.io.gfile.makedirs(os.path.dirname(path))

    with tf.io.TFRecordWriter(path) as writer:
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "observations/images0": tensor_feature(
                        np.array(
                            [o["image"] for o in trajectory["observations"]],
                            dtype=np.uint8,
                        )
                    ),
                    "observations/state": tensor_feature(
                        np.array(
                            [o["proprio"] for o in trajectory["observations"]],
                            dtype=np.float32,
                        )
                    ),
                    "next_observations/images0": tensor_feature(
                        np.array(
                            [o["image"] for o in trajectory["next_observations"]],
                            dtype=np.uint8,
                        )
                    ),
                    "next_observations/state": tensor_feature(
                        np.array(
                            [o["proprio"] for o in trajectory["next_observations"]],
                            dtype=np.float32,
                        )
                    ),
                    "actions": tensor_feature(
                        np.array(trajectory["actions"], dtype=np.float32)
                    ),
                    **(
                        {
                            "rewards": tensor_feature(
                                np.array(trajectory["rewards"], dtype=np.float32)
                            ),
                            "masks": tensor_feature(
                                np.array(trajectory["masks"], dtype=np.float32)
                            ),
                            "mc_returns": tensor_feature(
                                np.array(trajectory["mc_returns"], dtype=np.float32)
                            ),
                        }
                        if "rewards" in trajectory
                        else {}
                    ),
                }
            )
        )
        writer.write(example.SerializeToString())
