"""
tf.data.Dataset based dataloader for Calvin dataset.
"""
from typing import Optional, Tuple
import click
import os
import math
from functools import partial
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import multiprocessing as mp

# from concurrent.futures import ProcessPoolExecutor
from multiprocess import Pool, cpu_count
from jaxrl_m.utils.train_utils import tensor_feature
from jaxrl_m.common.traj import calc_return_to_go
from jaxrl_m.envs.calvin import CalvinEnv, get_calvin_config, get_calvin_env


CALVIN_EVAL_RESET_STATE = (
    np.array(
        [
            -0.07788351,
            -0.12043487,
            0.57556621,
            -3.13497289,
            -0.03881178,
            1.57496315,
            0.07999775,
            -0.95831307,
            0.91764932,
            2.14034334,
            -2.33283112,
            -0.73032276,
            1.71993711,
            0.73225109,
            1.0,
        ]
    ),
    np.array(
        [
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            3.21791205e-17,
            0.00000000e00,
            0.00000000e00,
            -2.02787788e-01,
            7.80806667e-02,
            4.60989300e-01,
            -4.12960108e-05,
            5.98985237e-06,
            -1.60058229e00,
            -2.87344400e-01,
            -8.21375894e-02,
            4.59983870e-01,
            3.33259549e-05,
            3.60273817e-04,
            -1.23889334e00,
            6.13612809e-02,
            -1.36545894e-01,
            4.59990000e-01,
            8.97646026e-07,
            -6.92398241e-08,
            2.85597653e00,
        ]
    ),
)


def convert_calvin_chunk_to_tfrecord(
    dataset_path: str,
    output_path: str,
    image_key: str,
    action_key: str,
    image_size: int,
    start_index: int,
    end_index: int,
    chunk_index: int,
    include_next_observations: bool,
    only_states: bool = False,
    states_with_distractors: bool = False,
    rerender_images_on_cpu: bool = False,
    env: Optional[CalvinEnv] = None,
    include_rewards: bool = False,
    reward_bias: float = -4.0,
    rewards: Optional[np.ndarray] = None,
    masks: Optional[np.ndarray] = None,
    mc_returns: Optional[np.ndarray] = None,
):
    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")
    if not only_states:
        images = np.zeros(
            (end_index - start_index + 1, image_size, image_size, 3), dtype=np.uint8
        )
    else:
        images = None
    states_dim = 21 if not states_with_distractors else 39
    states = np.zeros((end_index - start_index + 1, states_dim), dtype=np.float32)
    actions = np.zeros((end_index - start_index + 1, 7), dtype=np.float32)
    if include_rewards:
        assert rewards.shape == (end_index - start_index + 1,)
        assert masks.shape == (end_index - start_index + 1,)
        assert mc_returns.shape == (end_index - start_index + 1,)

    for i in range(start_index, end_index + 1):
        transition = np.load(os.path.join(dataset_path, f"episode_{i:07d}.npz"))

        if not only_states:
            if rerender_images_on_cpu:
                env.reset_to_state(
                    robot_obs=transition["robot_obs"], scene_obs=transition["scene_obs"]
                )
                image = env.render(mode="rgb_array")
            else:
                image = transition[image_key]
                # Resize image
                image = cv2.resize(image, (image_size, image_size))
            images[i - start_index] = image
        states[i - start_index] = np.concatenate(
            [transition["robot_obs"], transition["scene_obs"]]
        )[:states_dim]
        actions[i - start_index] = transition[action_key]

    episode_path = os.path.join(output_path, f"episode_{chunk_index}.tfrecord")
    # assert not tf.io.gfile.exists(episode_path)
    if tf.io.gfile.exists(episode_path):
        # Overwrite existing file
        tf.io.gfile.remove(episode_path)
        print(f"Overwriting existing episode {episode_path}")
    tf.io.gfile.makedirs(os.path.dirname(episode_path))

    observation_features = (
        {
            "observations/state": tensor_feature(states),
        }
        if not include_next_observations
        else {
            "observations/state": tensor_feature(states[:-1]),
            "next_observations/state": tensor_feature(states[1:]),
        }
    )
    if not only_states:
        observation_features.update(
            {
                "observations/images0": tensor_feature(images),
            }
            if not include_next_observations
            else {
                "observations/images0": tensor_feature(images[:-1]),
                "next_observations/images0": tensor_feature(images[1:]),
            }
        )
    if include_rewards:
        observation_features["rewards"] = tensor_feature(rewards[1:])
        observation_features["masks"] = tensor_feature(masks[1:])
        observation_features["mc_returns"] = tensor_feature(mc_returns[1:])

    with tf.io.TFRecordWriter(episode_path) as writer:
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    **observation_features,
                    "actions": tensor_feature(actions[:-1]),
                }
            )
        )
        writer.write(example.SerializeToString())

    print(f"Finished writing episode {chunk_index}")


@click.command()
@click.option(
    "--dataset_path",
    type=str,
    default="/iris/u/maxsobolmark/calvin/dataset/task_D_D/training/",
)
@click.option(
    "--output_path",
    type=str,
    default="/iris/u/maxsobolmark/calvin/dataset/task_D_D/training_tfrecords/",
)
@click.option("--image_key", type=str, default="rgb_static")
@click.option("--action_key", type=str, default="rel_actions")
@click.option("--image_size", type=int, default=100)
@click.option("--max_episode_length", type=int, default=500)
@click.option("--include_next_observations", type=bool, default=False)
@click.option("--only_states", type=bool, default=False)
@click.option("--states_with_distractors", type=bool, default=False)
@click.option("--rerender_images_on_cpu", type=bool, default=False)
@click.option("--include_rewards", type=bool, default=False)
@click.option("--reward_bias", type=float, default=-4.0)
@click.option("--only_process_episode_index", type=int, default=None)
def convert_calvin_dataset_to_tfrecord(
    dataset_path: str = "/iris/u/maxsobolmark/calvin/dataset/task_D_D/training/",
    output_path: str = "/iris/u/maxsobolmark/calvin/dataset/task_D_D/training_tfrecords/",
    image_key: str = "rgb_static",
    action_key: str = "rel_actions",
    image_size: int = 100,
    max_episode_length: int = 500,  # Default max episode length, will produce 499 transitions
    include_next_observations: bool = False,
    only_states: bool = False,
    states_with_distractors: bool = False,
    rerender_images_on_cpu: bool = False,
    include_rewards: bool = False,
    reward_bias: float = -4.0,
    only_process_episode_index: Optional[int] = None,
):
    assert dataset_path.endswith("task_D_D/training/")
    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")
    start_end_ids = np.load(os.path.join(dataset_path, "ep_start_end_ids.npy"))

    episode_counter = 0  # To keep track of new episode indices

    if rerender_images_on_cpu:
        calvin_config = get_calvin_config()
        calvin_config["cameras"]["static"]["width"] = image_size
        calvin_config["cameras"]["static"]["height"] = image_size
        calvin_config["screen_size"] = [image_size, image_size]
        calvin_config["use_egl"] = False
        env = get_calvin_env(cfg=calvin_config)
        env.reset()
    else:
        env = None

    # with ProcessPoolExecutor(max_workers=4) as executor:
    # with Pool(processes=cpu_count(), maxtasksperchild=1) as pool:
    futures = []
    for episode_index, (start_index, end_index) in tqdm(enumerate(start_end_ids)):
        num_transitions = end_index - start_index + 1
        if include_rewards and (
            only_process_episode_index is None
            or only_process_episode_index == episode_index
        ):
            (
                rewards,
                masks,
                mc_returns,
            ) = calculate_rewards_masks_and_return_to_go_for_episode(
                dataset_path, start_index, end_index, reward_bias, env
            )
        else:
            rewards = masks = mc_returns = np.zeros(num_transitions)

        # Split the episode into chunks of max_episode_length, except the last chunk
        num_chunks = math.ceil(num_transitions / max_episode_length)

        for chunk_index in tqdm(range(num_chunks), desc="chunk progress", leave=False):
            chunk_start = chunk_index * max_episode_length
            chunk_end = min((chunk_index + 1) * max_episode_length, num_transitions)

            # futures.append(
            #     # executor.submit(
            #     pool.apply_async(
            #         partial(
            #             convert_calvin_chunk_to_tfrecord,
            #             dataset_path,
            #             output_path,
            #             image_key,
            #             action_key,
            #             image_size,
            #             start_index + chunk_start,
            #             start_index + chunk_end - 1,
            #             episode_counter,
            #             include_next_observations,
            #             only_states,
            #             states_with_distractors,
            #             rerender_images_on_cpu,
            #             env,
            #             include_rewards,
            #             reward_bias,
            #             rewards[chunk_start:chunk_end],
            #             masks[chunk_start:chunk_end],
            #             mc_returns[chunk_start:chunk_end],
            #         )
            #     )
            # )
            if (
                only_process_episode_index is None
                or only_process_episode_index == episode_index
            ):
                convert_calvin_chunk_to_tfrecord(
                    dataset_path,
                    output_path,
                    image_key,
                    action_key,
                    image_size,
                    start_index + chunk_start,
                    start_index + chunk_end - 1,
                    episode_counter,
                    include_next_observations,
                    only_states,
                    states_with_distractors,
                    rerender_images_on_cpu,
                    env,
                    include_rewards,
                    reward_bias,
                    rewards[chunk_start:chunk_end],
                    masks[chunk_start:chunk_end],
                    mc_returns[chunk_start:chunk_end],
                )

            episode_counter += 1

        # for future in tqdm(futures, desc="waiting for futures"):
        #     future.get()  # Wait for all futures to finish


# def get_obs_for_goal(
#     goal=np.array([0.25, 0.15, 0, 0.088, 1, 1]),
#     dataset_path: str = "/iris/u/maxsobolmark/calvin/dataset/task_D_D/training/",
# ):
#     min_distance = np.inf
#     min_index = -1
#     transition_paths = tf.io.gfile.glob(os.path.join(dataset_path, "episode_*.npz"))

#     for transition_path in tqdm(transition_paths):
#         transition = np.load(transition_path)
#         obs = transition["scene_obs"][:6]

#         distance = np.linalg.norm(obs - goal)
#         if distance < min_distance:
#             min_distance = distance
#             min_index = int(transition_path.split("/")[-1].split("_")[-1].split(".")[0])

#     return min_index, min_distance


def get_obs_for_goal_process_chunk(chunk, goal):
    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")
    min_distance = np.inf
    min_index = -1

    for transition_path in chunk:
        transition = np.load(transition_path)
        obs = transition["scene_obs"][:6]

        distance = np.linalg.norm(obs - goal)
        if distance < min_distance:
            min_distance = distance
            min_index = int(transition_path.split("/")[-1].split("_")[-1].split(".")[0])

    print(
        f"Finished processing chunk. Min index: {min_index}, Min distance: {min_distance}"
    )
    return min_index, min_distance


def get_obs_for_goal(
    goal=np.array([0.25, 0.15, 0, 0.088, 1, 1]),
    dataset_path: str = "/iris/u/maxsobolmark/calvin/dataset/task_D_D/training/",
    num_processes: int = mp.cpu_count(),
):
    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")
    transition_paths = tf.io.gfile.glob(os.path.join(dataset_path, "episode_*.npz"))

    # Split the transition_paths into chunks
    chunk_size = int((len(transition_paths) / 20) // num_processes)
    print(f"Chunk size: {chunk_size}")
    chunks = [
        transition_paths[i : i + chunk_size]
        for i in range(0, len(transition_paths), chunk_size)
    ]

    # Create a partial function with the goal parameter
    process_chunk_with_goal = partial(get_obs_for_goal_process_chunk, goal=goal)

    # Process chunks in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = list(
            tqdm(pool.imap(process_chunk_with_goal, chunks), total=len(chunks))
        )

    # Find the overall minimum
    min_index, min_distance = min(results, key=lambda x: x[1])

    # (299826, 0.001168673166969729)
    return min_index, min_distance


def create_goal_image_file(
    transition_index: int,
    output_path: str = "/iris/u/maxsobolmark/calvin/dataset/task_D_D/goal_image.npz",
    image_size: int = 100,
):
    transition_path = f"/iris/u/maxsobolmark/calvin/dataset/task_D_D/training/episode_{transition_index:07d}.npz"
    transition = np.load(transition_path)
    original_goal_image = transition["rgb_static"]
    original_goal_image = cv2.resize(original_goal_image, (image_size, image_size))
    # Make image by actually rendering in the environment
    calvin_config = get_calvin_config()
    calvin_config["screen_size"] = [image_size, image_size]
    env = get_calvin_env(cfg=calvin_config, goal_sampler=original_goal_image[None])
    env.reset_to_state(
        robot_obs=transition["robot_obs"], scene_obs=transition["scene_obs"]
    )
    env_goal_image = env.render(mode="rgb_array")
    env_goal_image = cv2.resize(env_goal_image, (image_size, image_size))

    np.savez(output_path, goal_image=original_goal_image)
    np.savez(output_path.replace(".npz", "_env.npz"), goal_image=env_goal_image)
    # Also save the goal image as a PNG
    cv2.imwrite(output_path.replace(".npz", ".png"), original_goal_image)
    cv2.imwrite(output_path.replace(".npz", "_env.png"), env_goal_image)


def calculate_rewards_masks_and_return_to_go_for_episode(
    dataset_path: str,
    episode_start_index: int,
    episode_end_index: int,
    reward_bias: float,
    env: CalvinEnv,
    discount_factor: float = 0.99,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert env.markovian_rewards, "This function only works with markovian rewards"
    rewards = np.zeros((episode_end_index - episode_start_index + 1,), dtype=np.float32)
    env.unwrapped.reset()
    env_start_info = env.unwrapped.get_info()

    for i in tqdm(
        range(episode_start_index, episode_end_index + 1),
        desc="Calculating rewards for episode",
        leave=False,
    ):
        transition_path = os.path.join(dataset_path, f"episode_{i:07d}.npz")
        transition = np.load(transition_path)
        env.unwrapped.reset_to_state(
            robot_obs=transition["robot_obs"], scene_obs=transition["scene_obs"]
        )
        env.unwrapped.start_info = env_start_info
        reward, _ = env.unwrapped._reward()
        rewards[i - episode_start_index] = reward + reward_bias

    if reward_bias <= -4.0:
        # rewards are always negative/0, so terminate on complete success
        masks = (rewards != 4.0 + reward_bias).astype(np.float32)
    else:
        masks = np.ones_like(rewards, dtype=np.float32)

    return_to_go = calc_return_to_go(
        rewards, masks, discount_factor, push_failed_to_min=False
    )
    return_to_go = np.array(return_to_go, dtype=np.float32)

    return rewards, masks, return_to_go


if __name__ == "__main__":
    convert_calvin_dataset_to_tfrecord()
