from typing import List
import os
from glob import glob
import click
import time
import numpy as np
import tensorflow as tf
import torch
import pickle
from jaxrl_m.agents.continuous.openvla import OpenVLAAgent
from jaxrl_m.utils.timer_utils import Timer
from jaxrl_m.common.wandb import WandBLogger
import wandb

PROTO_TYPE_SPEC = {
    "observations/images0": tf.uint8,
    "next_observations/images0": tf.uint8,
}


def tfrecord_to_dict(tfrecord_path, proto_type_spec=PROTO_TYPE_SPEC):
    raw_dataset = tf.data.TFRecordDataset([tfrecord_path])
    features = {
        key: tf.io.FixedLenFeature([], tf.string) for key in proto_type_spec.keys()
    }
    for raw_record in raw_dataset.take(1):
        parsed_features = tf.io.parse_single_example(raw_record, features)
    parsed_tensors = {
        key: tf.io.parse_tensor(parsed_features[key], dtype)
        for key, dtype in proto_type_spec.items()
    }
    return parsed_tensors


# python scripts/openvla_caching_worker.py
# --checkpoints_dir=./results/PA-RL/ft_openvla/seed_0/base_policy_checkpoints_from_agent_trainer
# --instruction="put eggplant on yellow plate" --repeat=8


@click.command()
@click.option("--checkpoints_dir", type=str)
@click.option("--instruction", type=str)
@click.option("--repeat", type=int)
@click.option("--action_std", type=float, default=0.1)
@click.option("--worker_id", type=int, default=0)
@click.option("--num_workers", type=int, default=1)
def openvla_caching_worker(
    checkpoints_dir: str,
    instruction: str,
    repeat: int,
    action_std: float = 0.1,
    worker_id: int = 0,
    num_workers: int = 1,
):
    os.umask(0)
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    os.environ["WANDB_INIT_TIMEOUT"] = "120"
    wandb.require("core")
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": "PA-RL",
            "exp_descriptor": f"openvla_caching_worker_{worker_id}/{num_workers}",
            "tag": None,
        }
    )
    # wandb is useful to see which worker indices are running, even if we don't log anything.
    wandb_logger = WandBLogger(  # noqa
        wandb_config=wandb_config,
        debug=False,
        variant=dict(
            checkpoints_dir=checkpoints_dir,
            instruction=instruction,
            repeat=repeat,
            action_std=action_std,
            worker_id=worker_id,
            num_workers=num_workers,
        ),
    )

    timer = Timer()
    action_std = [action_std] * 7
    assert checkpoints_dir.endswith("base_policy_checkpoints_from_agent_trainer")

    openvla_agent = OpenVLAAgent.create(
        action_std=action_std,
        instruction=instruction,
    )
    current_checkpoint_index = -1

    episodes_left_to_cache: List[str] = []

    while True:
        available_checkpoints = glob(os.path.join(checkpoints_dir, "checkpoint_*"))
        available_checkpoint_indices = [
            int(checkpoint.split("_")[-1]) for checkpoint in available_checkpoints
        ]
        available_checkpoint_indices.sort()
        if available_checkpoint_indices:
            latest_checkpoint_index = available_checkpoint_indices[-1]
            # Check if latest checkpoint is ready to use
            # I.e. there's two dirs, openvla_processor and openvla_model
            fully_formed_checkpoint = all(
                [
                    os.path.exists(
                        os.path.join(
                            checkpoints_dir,
                            f"checkpoint_{latest_checkpoint_index}",
                            dir_name,
                        )
                    )
                    for dir_name in ["openvla_processor", "openvla_model"]
                ]
            )
            # and adapter_config.json is in openvla_model
            fully_formed_checkpoint = (
                fully_formed_checkpoint
                and os.path.exists(
                    os.path.join(
                        checkpoints_dir,
                        f"checkpoint_{latest_checkpoint_index}",
                        "openvla_model",
                        "adapter_config.json",
                    )
                )
                and os.path.exists(
                    os.path.join(
                        checkpoints_dir,
                        f"checkpoint_{latest_checkpoint_index}",
                        "openvla_model",
                        "adapter_model.safetensors",
                    )
                )
            )

            if (
                latest_checkpoint_index != current_checkpoint_index
                and fully_formed_checkpoint
            ):
                current_checkpoint_index = latest_checkpoint_index
                print(
                    f"Worker {worker_id} is loading checkpoint {current_checkpoint_index}"
                )
                openvla_agent.load_checkpoint(
                    os.path.join(
                        checkpoints_dir, f"checkpoint_{current_checkpoint_index}"
                    )
                )
                episodes_left_to_cache = []
                # Re-calculate the episodes left to cache
                # image replay buffer lives in the parent directory of the checkpoints
                image_replay_buffer_path = os.path.join(
                    checkpoints_dir, "..", "image_replay_buffer"
                )
                all_episodes = glob(
                    os.path.join(image_replay_buffer_path, "*.tfrecord")
                )
                # Get the onces that are worker_id modulo num_workers
                episodes_left_to_cache = [
                    episode
                    for episode in all_episodes
                    if int(episode.split("_")[-1].split(".")[0]) % num_workers
                    == worker_id
                ]

        if current_checkpoint_index != -1 and len(episodes_left_to_cache) > 0:
            episode_path = episodes_left_to_cache.pop(0)
            print(
                f"Worker {worker_id} is caching episode {episode_path} ({len(episodes_left_to_cache)} left)"
            )

            episode = tfrecord_to_dict(episode_path)
            # Put all the images in a numpy array
            observations_images = episode["observations/images0"].numpy()
            assert len(observations_images.shape) == 4
            next_observations_images = episode["next_observations/images0"].numpy()
            assert len(next_observations_images.shape) == 4
            all_observation_images = np.concatenate(
                [observations_images, next_observations_images], axis=0
            )
            # Get gpu memory size in gb
            gpu_memory_size_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory_size_gb > 48:
                openvla_agent.sample_actions(
                    all_observation_images,
                    repeat=repeat,
                    timer=timer,
                    wait_for_cache=False,
                )
            else:
                # Split the images into chunks
                chunk_size = 16
                for i in range(0, len(all_observation_images), chunk_size):
                    openvla_agent.sample_actions(
                        all_observation_images[i : i + chunk_size],
                        repeat=repeat,
                        timer=timer,
                        wait_for_cache=False,
                    )

            # Save cache
            cache_save_path = os.path.join(
                checkpoints_dir,
                f"checkpoint_{current_checkpoint_index}",
                "openvla_cache",
                os.path.basename(episode_path).replace(".tfrecord", ".pkl"),
            )

            os.makedirs(os.path.dirname(cache_save_path), exist_ok=True)
            cache = openvla_agent.action_cache
            with open(cache_save_path, "wb") as f:
                pickle.dump(cache, f)
            openvla_agent.clear_cache()

        else:
            print(f"Worker {worker_id} is waiting for new checkpoints")

        time.sleep(0.5)


if __name__ == "__main__":
    openvla_caching_worker()
