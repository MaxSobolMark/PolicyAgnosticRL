import os
from typing import List
import click
from tqdm import tqdm
from metaworld import MT1
import numpy as np
import jax
import cv2
import pickle
from jaxrl_m.envs.metaworld import get_scripted_policy
from jaxrl_m.common.traj import calc_return_to_go


@click.command()
@click.option("--task_name", type=str, required=True)
@click.option("--num_trajectories", type=int, default=5)
@click.option("--output_path", type=str, default="metaworld_datasets")
@click.option("--seed", type=int, default=0)
@click.option("--camera_name", type=str, default="topview")
@click.option("--reward_bias", type=float, default=-1.0)
@click.option("--save_video", is_flag=True)
def create_metaworld_dataset(
    task_name: str,
    num_trajectories: int = 5,
    output_path: str = "metaworld_datasets",
    seed: int = 0,
    camera_name: str = "topview",
    reward_bias: float = -1.0,
    save_video: bool = False,
):
    assert (
        num_trajectories <= 50
    ), "There are only 50 possible initial conditions for each task."
    # Add MUJOCO_GL=egl to the environment variables to use EGL rendering
    os.environ["MUJOCO_GL"] = "egl"
    mt1 = MT1(task_name, seed=seed)
    env = mt1.train_classes[task_name](render_mode="rgb_array", camera_name=camera_name)
    policy = get_scripted_policy(task_name)

    observations = []
    next_observations = []
    actions = []
    rewards = []
    masks = []
    mc_returns = []
    frames: List[List[np.ndarray]] = []
    max_trajectory_length = 0

    for trajectory_index in tqdm(range(num_trajectories)):
        env.set_task(mt1.train_tasks[trajectory_index])
        obs, _ = env.reset()
        # observations.append(obs)
        trajectory_observations = [obs]
        trajectory_actions = []
        trajectory_rewards = []
        frames.append([])
        frames[-1].append(env.render()[..., ::-1])
        done = False
        while not done:
            action = policy.sample_actions(obs[None, None])[0][0]
            action = jax.device_get(action)
            trajectory_actions.append(action)
            obs, rew, terminated, truncated, info = env.step(action)
            trajectory_rewards.append(info["success"])
            trajectory_observations.append(obs)
            frame = env.render()[..., ::-1]
            # If reward is 1, add green border
            if info["success"] == 1.0:
                frame[:10, :, :] = [0, 255, 0]
                frame[-10:, :, :] = [0, 255, 0]
                frame[:, :10, :] = [0, 255, 0]
                frame[:, -10:, :] = [0, 255, 0]
            done = (info["success"] == 1.0) or terminated or truncated
            frames[-1].append(frame)

        max_trajectory_length = max(max_trajectory_length, len(frames[-1]))
        observations.append(np.array(trajectory_observations)[:-1])
        next_observations.append(np.array(trajectory_observations)[1:])
        actions.append(np.array(trajectory_actions))
        trajectory_rewards = np.array(trajectory_rewards)
        trajectory_rewards = trajectory_rewards + reward_bias
        rewards.append(trajectory_rewards)
        trajectory_masks = (trajectory_rewards != (1.0 + reward_bias)).astype(
            np.float32
        )
        masks.append(trajectory_masks)
        trajectory_mc_returns = calc_return_to_go(
            trajectory_rewards, trajectory_masks, 0.99, push_failed_to_min=True
        )
        mc_returns.append(trajectory_mc_returns)

    observations = np.concatenate(observations, axis=0)
    next_observations = np.concatenate(next_observations, axis=0)
    actions = np.concatenate(actions, axis=0)
    rewards = np.concatenate(rewards, axis=0)
    masks = np.concatenate(masks, axis=0)
    mc_returns = np.concatenate(mc_returns, axis=0)

    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f"{output_path}/{task_name}", exist_ok=True)

    # Save dataset
    with open(f"{output_path}/{task_name}/{task_name}.pkl", "wb") as f:
        pickle.dump(
            {
                "observations": observations,
                "next_observations": next_observations,
                "actions": actions,
                "rewards": rewards,
                "masks": masks,
                "mc_returns": mc_returns,
            },
            f,
        )

    if not save_video:
        print(f"Dataset saved at {output_path}/{task_name}")
        return
    # Create mp4 putting all trajectories in a row
    # If a trajectory is shorter than the longest one, pad it with its last frame
    for trajectory_frames in frames:
        trajectory_frames += [trajectory_frames[-1]] * (
            max_trajectory_length - len(trajectory_frames)
        )
    out = cv2.VideoWriter(
        f"{output_path}/{task_name}/{task_name}.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (480 * num_trajectories, 480),
    )
    for frame_index in range(max_trajectory_length):
        out.write(
            np.concatenate([frame[frame_index] for frame in frames], axis=1).astype(
                np.uint8
            )
        )
    out.release()
    print(f"Dataset saved at {output_path}/{task_name}")


if __name__ == "__main__":
    create_metaworld_dataset()
