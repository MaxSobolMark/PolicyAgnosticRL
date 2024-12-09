from typing import List

import numpy as np
import tensorflow as tf

from jaxrl_m.data.tf_augmentations import augment
from jaxrl_m.data.tf_goal_relabeling import GOAL_RELABELING_FUNCTIONS


class RoboverseDataset:
    """models after BridgeDataset to load Roboverse Data"""

    def __init__(
        self,
        data_paths,
        batch_size,
        reward_scale,
        reward_bias,
        seed,
        augment=True,
        augment_kwargs={},
        augment_next_obs_goal_differently=False,
        train=True,
        get_mc_return=False,
        make_goal_conditioned_dataset=False,
        goal_relabeling_strategy: str = "uniform",
        goal_relabeling_kwargs: dict = {},
        discount=0.96,
        sample_weights=None,
        shuffle_buffer_size=10000,
    ):
        if isinstance(data_paths[0], str):
            data_paths = [data_paths]
        if sample_weights is None:
            # default to uniform distribution over sub-lists
            sample_weights = [1 / len(data_paths)] * len(data_paths)

        self.batch_size = batch_size
        self.reward_scale = reward_scale
        self.reward_bias = reward_bias
        self.augment_kwargs = augment_kwargs
        self.augment_next_obs_goal_differently = augment_next_obs_goal_differently
        self.get_mc_return = get_mc_return
        self.make_goal_conditioned_dataset = make_goal_conditioned_dataset
        self.goal_relabeling_strategy = goal_relabeling_strategy
        self.goal_relabeling_kwargs = goal_relabeling_kwargs
        self.discount = discount

        datasets = []
        for sub_data_paths in data_paths:
            datasets.append(self._construct_tf_dataset(sub_data_paths, seed))

        if train:
            # shuffle and repeat each sub-dataset, allocating the shuffle buffer
            # by sample_weights
            for i in range(len(datasets)):
                datasets[i] = (
                    datasets[i]
                    .shuffle(int(shuffle_buffer_size * sample_weights[i]), seed + i)
                    .repeat()
                )

        # for validation, we want to be able to iterate through the entire dataset;
        # for training, we want to make sure that no sub-dataset is ever exhausted
        # or the sampling ratios will be off. this should never happen because of the
        # repeat() above, but `stop_on_empty_dataset` is a safeguard
        dataset = tf.data.Dataset.sample_from_datasets(
            datasets, sample_weights, seed=seed, stop_on_empty_dataset=train
        )

        if train and augment:
            # apply augmentations, using a sequence of integers as seeds.
            # this was the only way I found to avoid a memory leak in tf.random.Generator
            dataset = dataset.enumerate(start=seed)
            dataset = dataset.map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.batch(
            batch_size,
            num_parallel_calls=tf.data.AUTOTUNE,
            drop_remainder=True,
            deterministic=not train,
        )

        self.tf_dataset = dataset

    def _construct_tf_dataset(self, paths: List[str], seed: int) -> tf.data.Dataset:
        """
        Constructs a tf.data.Dataset from a list of paths.
        The dataset yields a dictionary of tensors for each transition.
        """

        # shuffle again using the dataset API so the files are read in a
        # different order every epoch
        dataset = tf.data.Dataset.from_tensor_slices(paths).shuffle(len(paths), seed)

        # yields raw serialized examples
        dataset = tf.data.TFRecordDataset(dataset, num_parallel_reads=tf.data.AUTOTUNE)

        # yields trajs
        dataset = dataset.map(self._decode_example, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self._add_masks, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(
            self._process_rewards, num_parallel_calls=tf.data.AUTOTUNE
        )
        if self.make_goal_conditioned_dataset:
            dataset = dataset.map(self._add_goals, num_parallel_calls=tf.data.AUTOTUNE)

        # unbatch to yield individual transitions
        dataset = dataset.unbatch()

        return dataset

    PROTO_TYPE_SPEC = {
        "observations/images0": tf.uint8,
        "observations/state": tf.float32,
        "next_observations/images0": tf.uint8,
        "next_observations/state": tf.float32,
        "actions": tf.float32,
        "terminals": tf.bool,
        "truncates": tf.bool,
        "rewards": tf.float32,
    }

    def _decode_example(self, example_proto):
        # decode the example proto according to PROTO_TYPE_SPEC
        features = {
            key: tf.io.FixedLenFeature([], tf.string)
            for key in self.PROTO_TYPE_SPEC.keys()
        }
        parsed_features = tf.io.parse_single_example(example_proto, features)
        parsed_tensors = {
            key: tf.io.parse_tensor(parsed_features[key], dtype)
            for key, dtype in self.PROTO_TYPE_SPEC.items()
        }
        # restructure the dictionary into the downstream format
        return {
            "observations": {
                "image": parsed_tensors["observations/images0"],
                "proprio": parsed_tensors["observations/state"],
            },
            "next_observations": {
                "image": parsed_tensors["next_observations/images0"],
                "proprio": parsed_tensors["next_observations/state"],
            },
            "actions": parsed_tensors["actions"],
            "rewards": parsed_tensors["rewards"],
            "terminals": parsed_tensors["terminals"],
            "truncates": parsed_tensors["truncates"],
        }

    def _process_rewards(self, traj):
        traj["rewards"] = traj["rewards"] * self.reward_scale + self.reward_bias

        if self.get_mc_return:
            traj["mc_returns"] = calc_return_to_go(
                traj["rewards"], traj["masks"], gamma=self.discount
            )

        return traj

    def _add_masks(self, traj):
        traj["masks"] = tf.math.logical_not(traj["truncates"])
        return traj

    def _add_goals(self, traj):
        traj = GOAL_RELABELING_FUNCTIONS[self.goal_relabeling_strategy](
            traj, **self.goal_relabeling_kwargs
        )

        return traj

    def _augment(self, seed, transition):

        to_augment = ["observations", "next_observations"]
        if self.make_goal_conditioned_dataset:
            to_augment.append("goals")
        n_aug = len(to_augment)

        if self.augment_next_obs_goal_differently:
            sub_seeds = tf.unstack(
                tf.random.stateless_uniform(
                    [n_aug, 2],
                    seed=[seed, seed],
                    minval=None,
                    maxval=None,
                    dtype=tf.int32,
                )
            )
        else:
            # use the same seed for obs, next_obs, [and goal]
            sub_seeds = [[seed, seed]] * n_aug

        for key, sub_seed in zip(to_augment, sub_seeds):
            transition[key]["image"] = augment(
                transition[key]["image"], sub_seed, **self.augment_kwargs
            )
        return transition

    def iterator(self):
        return self.tf_dataset.prefetch(tf.data.AUTOTUNE).as_numpy_iterator()


def calc_return_to_go(rewards, masks, gamma):
    """
    This only works on put-ball-in-bowl task, which has sparse rewards
    all inputs are tf.tensors
    """

    def failure_return_to_go():
        return rewards[-1] / (1 - gamma) * tf.ones_like(rewards, dtype=tf.float32)

    def success_return_to_go():
        return_to_go = tf.TensorArray(dtype=tf.float32, size=len(rewards))
        prev_return = tf.TensorArray(dtype=tf.float32, size=1)
        prev_return = prev_return.write(0, 0.0)
        for i in tf.range(len(rewards)):
            index = len(rewards) - i - 1
            return_to_go = return_to_go.write(
                index,
                rewards[index]
                + gamma * prev_return.read(0) * tf.cast(masks[index], tf.float32),
            )
            prev_return = prev_return.write(0, return_to_go.read(index))

        return return_to_go.stack()

    return tf.cond(
        tf.reduce_all(rewards < 0), failure_return_to_go, success_return_to_go
    )


def numpy_datafile_to_tf_records(np_data_path):
    """converts the saved out.npy to out.tfrecords"""

    print("converting numpy datafile to tf records: ", np_data_path)

    def tensor_feature(value):
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()])
        )

    # read in the numpy file
    data = np.load(np_data_path, allow_pickle=True)

    out_path = np_data_path.replace(".npy", ".tfrecord")
    with tf.io.TFRecordWriter(out_path) as writer:
        for traj in data:
            truncates = np.zeros(len(traj["actions"]), dtype=np.bool_)
            truncates[-1] = True
            steps_existing = np.arange(len(traj["actions"]), dtype=np.int32)
            steps_remaining = steps_existing[::-1]

            infos = {}
            for key in data[0]["env_infos"][0]:
                infos[f"infos/{key}"] = tensor_feature(
                    np.array([i[key] for i in traj["env_infos"]])
                )

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "observations/images0": tensor_feature(
                            np.array(
                                [o["image"] for o in traj["observations"]],
                                dtype=np.uint8,
                            )
                        ),
                        "observations/state": tensor_feature(
                            np.array(
                                [o["state"] for o in traj["observations"]],
                                dtype=np.float32,
                            )
                        ),
                        "next_observations/images0": tensor_feature(
                            np.array(
                                [o["image"] for o in traj["next_observations"]],
                                dtype=np.uint8,
                            )
                        ),
                        "next_observations/state": tensor_feature(
                            np.array(
                                [o["state"] for o in traj["next_observations"]],
                                dtype=np.float32,
                            )
                        ),
                        "actions": tensor_feature(
                            np.array(traj["actions"], dtype=np.float32)
                        ),
                        "terminals": tensor_feature(
                            np.zeros(len(traj["actions"]), dtype=np.bool_)
                        ),
                        "truncates": tensor_feature(truncates),
                        "rewards": tensor_feature(
                            np.array(traj["rewards"], dtype=np.float32)
                        ),
                        "steps_existing": tensor_feature(steps_existing),
                        "steps_remaining": tensor_feature(steps_remaining),
                        **infos,
                    }
                )
            )
            writer.write(example.SerializeToString())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--np_data_path",
        type=str,
        default="/nfs/kun2/users/paulzhou/cog_data/minibullet/pickplacedata_noise0.1/PutBallintoBowl-v0/train/out.npy",
    )
    args = parser.parse_args()
    numpy_datafile_to_tf_records(args.np_data_path)
