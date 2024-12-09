######################################################
#                   maze2d_utils.py                  #
#       Code to generate goal missing datasets       #
######################################################

import os
from typing import Dict, Optional, Tuple
from collections import defaultdict

import d4rl
import gym
from d4rl.pointmaze import waypoint_controller
import numpy as np

from tqdm import tqdm
from matplotlib import pyplot as plt
from PIL import Image

from jaxrl_m.utils.timer_utils import Timer

import mujoco_py


def generate_downsampled_dataset(
    env_name, num_samples: int, save_path: Optional[str], random_seed: int = 0
) -> Dict[str, np.ndarray]:
    """
    Generate a downsampled dataset for a given environment.
    Args:
        env_name (str): Name of the environment.
        num_samples (int): Number of samples in the downsampled dataset.
        save_path (Optional[str]): Path to save the downsampled dataset.
        random_seed (int): Random seed.
    Returns:
        Dict[str, np.ndarray]: Downsampled dataset.
    """
    np.random.seed(random_seed)
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)
    if num_samples is None:
        assert save_path is None
        return dataset
    sampled_indices = np.random.choice(
        len(dataset["rewards"]), num_samples, replace=False
    )
    sampled_dataset = {}
    for key in dataset.keys():
        sampled_dataset[key] = dataset[key][sampled_indices]

    if save_path is not None:
        np.savez_compressed(save_path, **sampled_dataset)

    return sampled_dataset


def plot_toy_figure(
    dataset_path: str,
    save_path: str,
    env_name: str = "maze2d-large-v1",
    figsize: Tuple[int, int] = (5, 7),
    point_color: str = "r",
    reward_color: str = "g",
):

    env = gym.make(env_name)
    env.reset()

    plt.figure(figsize=figsize)

    env.viewer = mujoco_py.MjRenderContextOffscreen(env.sim, -1)
    env.viewer.cam.azimuth = -270
    env.viewer.cam.elevation = -90
    env.viewer.cam.distance = 20
    plt.imshow(env.sim.render(2000, 2800, mode="offscreen"))

    dataset = np.load(dataset_path)

    # Plots observations as points
    x = dataset["observations"][:, 0]
    y = dataset["observations"][:, 1]
    # If dataset["rewards"] is 1, color point green. Else, color point red.
    colors = [reward_color if r == 1 else point_color for r in dataset["rewards"]]

    plt.scatter(x, y, s=5, c=colors)

    # remove axis
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    # remove ticks
    plt.tick_params(
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False,
        bottom=False,
    )
    plt.savefig(save_path, dpi=600, transparent=False)
    env.close()


# Plot 1
def plot_maze2d_dataset(
    dataset_path: str,
    save_path: str,
    env_name: str = "maze2d-large-v1",
    circle_center: Tuple[float, float] = None,
    circle_radius: float = 0.5,
):
    """
    Plot a maze2d dataset observations and actions.
    Highlights target location with a green circle.
    Highlights circle location with a red circle.
    Args:
        dataset_path (str): Path to the dataset.
        save_path (str): Path to save the plot.
        env_name (str): Name of the environment.
        circle_radius (float): Radius of the circle.
        circle_center (Tuple[float, float]): Center of the circle.
    """
    plt.cla()
    dataset = np.load(dataset_path)

    # Plots observations as points
    x_positions = dataset["observations"][:, 0]
    y_positions = dataset["observations"][:, 1]
    plt.scatter(x_positions, y_positions, s=1)

    # Scatter plot dataset points
    x = dataset["observations"][:, 0] / 2
    y = dataset["observations"][:, 1]
    # If dataset["rewards"] is 1, color point green. Else, color point red.
    colors = ["g" if r == 1 else "r" for r in dataset["rewards"]]
    plt.scatter(x, y, s=1, c=colors)

    # Adds a green circle in the target location
    env = gym.make(env_name)
    target_location = plt.Circle(
        env.get_target(), 0.1, color="g", fill=False, linewidth=1
    )
    plt.gca().add_patch(target_location)

    # Adds a circle in given location, on top of the plot
    if circle_center is not None:
        circle = plt.Circle(
            circle_center, circle_radius, color="r", fill=False, linewidth=1
        )
        plt.gca().add_patch(circle)

    # Add x and y labels and title
    plt.xlabel("X Coordinates")
    plt.ylabel("Y Coordinates")

    # increase the resolution of the plot
    plt.savefig(save_path, dpi=300)


def plot_optimal_actions(
    dataset_path: str,
    save_path: str,
    env_name: str,
):
    """
    Plot the optimal actions for a maze2d dataset.
    Optimal actions are computed using the waypoint controller.
    Args:
        dataset_path (str): Path to the dataset.
        save_path (str): Path to save the plot.
        env_name (str): Name of the environment.
        target (Tuple[float, float]): Target location.
    """

    plt.cla()
    dataset = np.load(dataset_path)

    x_positions = dataset["observations"][:, 0]
    y_positions = dataset["observations"][:, 1]
    plt.scatter(x_positions, y_positions, s=1)

    # d_gain = -1.0
    # p_gain = 0.10
    env = gym.make(env_name)
    target = env.get_target()
    velocity = np.array((0, 0), dtype=np.float32)

    # For each observation, plot the optimal action as an arrow
    target_to_q_values = {}
    for i in tqdm(range(len(dataset["actions"]))):
        x = dataset["observations"][i, 0]
        y = dataset["observations"][i, 1]

        start = np.array((x, y), dtype=np.float32)
        controller = waypoint_controller.WaypointController(
            env.str_maze_spec,
            # p_gain=p_gain,
            # d_gain=d_gain,
            target_to_q_values=target_to_q_values,
        )
        target_to_q_values = controller.target_to_q_values
        act, done = controller.get_action(start, velocity, target)

        # Velocity is always (0, 0) so we can plot the action directly
        plt.arrow(
            x, y, act[0], act[1], color="r", head_width=0.0125, head_length=0.0125
        )

    # Adds a green circle in the target location
    target_location = plt.Circle(target, 0.1, color="g", fill=False, linewidth=1)
    plt.gca().add_patch(target_location)
    plt.savefig(save_path, dpi=300)


# Plot 2
def plot_points_within_circle(
    dataset_path: str,
    save_path: str,
    circle_center: Tuple[float, float],
    circle_radius: float,
    env_name: str = "maze2d-large-v1",
    random_seed: int = 0,
):
    """
    Plot the points within a circle in a maze2d dataset.
    Args:
        dataset_path (str): Path to the dataset.
        save_path (str): Path to save the plot.
        circle_center (Tuple[float, float]): Center of the circle.
        circle_radius (float): Radius of the circle.
    """
    plt.cla()
    plt.figure(figsize=(5, 7))
    dataset = np.load(dataset_path)

    env = gym.make(env_name)
    env.seed(random_seed)
    env.reset()
    np.random.seed(random_seed)
    velocity = np.array((0, 0), dtype=np.float32)
    # env.set_target((7,9))
    target = env.get_target()

    x_positions = dataset["observations"][:, 0]
    y_positions = dataset["observations"][:, 1]

    # Plot the points that are within the circle
    distances = np.sqrt(
        (x_positions - circle_center[0]) ** 2 + (y_positions - circle_center[1]) ** 2
    )
    within_circle = distances <= circle_radius
    plt.scatter(x_positions[within_circle], y_positions[within_circle], color="blue")

    # Plot the actions for the points within the circle
    for i in tqdm(range(len(x_positions[within_circle]))):
        x = x_positions[within_circle][i]
        y = y_positions[within_circle][i]

        controller = waypoint_controller.WaypointController(
            env.str_maze_spec,  # p_gain=p_gain, d_gain=d_gain
        )
        optimal_action, done = controller.get_action(np.array([x, y]), velocity, target)

        # Because velocity is taken into account we can't just plot the action directly
        # dx = dataset["next_observations"][:, 0][within_circle][i] - x
        # dy = dataset["next_observations"][:, 1][within_circle][i] - y
        # direction = np.array([dx, dy]) / np.linalg.norm([dx, dy]) * circle_radius * 0.9
        dx = dataset["actions"][:, 0][within_circle][i] * 0.1
        dy = dataset["actions"][:, 1][within_circle][i] * 0.1

        plt.arrow(
            x,
            y,
            dx,
            dy,
            # direction[0],
            # direction[1],
            color="black",
            head_width=0.0025,
            head_length=0.0025,
            linewidth=0.5,
        )
        # plt.arrow(
        #     x,
        #     y,
        #     optimal_action[0] * 0.1,
        #     optimal_action[1] * 0.1,
        #     color="red",
        #     head_width=0.0025,
        #     head_length=0.0025,
        #     linewidth=0.25,
        # )

    # Plot the circle center
    plt.scatter(circle_center[0], circle_center[1], color="red", marker="x")

    # Plot the circle boundary
    circle = plt.Circle(
        circle_center,
        circle_radius,
        color="red",
        fill=False,
        linestyle="--",
        label="Circle boundary",
    )
    plt.gca().add_patch(circle)

    # Plot the optimal action from the center of the circle
    # d_gain = -1.0
    # p_gain = 0.10
    start = np.array(circle_center, dtype=np.float32)
    controller = waypoint_controller.WaypointController(
        env.str_maze_spec,  # p_gain=p_gain, d_gain=d_gain
    )
    act, done = controller.get_action(start, velocity, target)
    # env.reset()
    # env.sim.reset()
    # env.set_state(start, velocity)
    # next_obs, _, _, _ = env.step(act)
    # optimal_delta = (
    #     (next_obs[:2] - start)
    #     / np.linalg.norm(next_obs[:2] - start)
    #     * circle_radius
    #     * 0.95
    # )

    # Velocity is always (0, 0) so we can plot the action directly
    plt.arrow(
        circle_center[0],
        circle_center[1],
        act[0] * 0.1,
        act[1] * 0.1,
        # optimal_delta[0],
        # optimal_delta[1],
        color="r",
        head_width=0.0025,
        head_length=0.0025,
        linewidth=0.5,
    )

    # Add x and y labels and title
    plt.xlabel("X Coordinates")
    plt.ylabel("Y Coordinates")
    plt.title("Maze2d zoomed-in with optimal action")
    # remove axis
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    # remove ticks
    plt.tick_params(
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False,
        bottom=False,
    )

    plt.savefig(save_path, dpi=300)


def generate_dataset_with_variable_optimality(
    noise_level: float,
    remove_optimal_actions: bool = False,
    optimality_threshold: float = 0.1,
    env_name: str = "maze2d-large-v1",
    num_samples: int = int(1e6),
    random_seed: int = 0,
    save_path: Optional[str] = None,
    remove_velocity: bool = False,
):
    """
    Generate a dataset with variable optimality.
    Args:
        env_name (str): Name of the environment.
        save_path (Optional[str]): Path to save the dataset.
        noise_level (float): Noise level.
        remove_optimal_actions (bool): Whether to remove optimal actions.
        optimality_threshold (float): Threshold for optimality.
    """

    new_dataset = {
        "observations": [],
        "actions": [],
        "terminals": [],
        "rewards": [],
        "next_observations": [],
    }

    from jaxrl_m.envs.pointmaze import PointMaze

    assert remove_velocity
    # env = gym.make(env_name)
    env = PointMaze("large")
    env.seed(random_seed)
    np.random.seed(random_seed)
    env.reset()

    target = env.get_target()
    dataset = d4rl.qlearning_dataset(env)
    del dataset["next_observations"]
    del dataset["rewards"]
    del dataset["terminals"]
    if remove_velocity:
        # Remove velocity from observations
        dataset["observations"] = dataset["observations"][:, :2]

    np.random.seed(random_seed)
    sampled_indices = np.random.choice(
        len(dataset["actions"]),
        num_samples,
        replace=False,
    )
    sampled_dataset = {}
    for key in dataset.keys():
        sampled_dataset[key] = dataset[key][sampled_indices]

    dataset = sampled_dataset

    target_to_q_values = {}

    # For each observation, compute noisy action
    for i in tqdm(range(len(dataset["actions"]))):
        position = np.array(dataset["observations"][i, 0:2], dtype=np.float32)
        if remove_velocity:
            velocity = np.zeros_like(position)
        else:
            velocity = np.array(dataset["observations"][i, 2:4], dtype=np.float32)

        # reset to state
        env.sim.reset()
        env.set_state(position, velocity)

        controller = waypoint_controller.WaypointController(
            env.str_maze_spec,
            # p_gain=p_gain,
            # d_gain=d_gain,
            target_to_q_values=target_to_q_values,
        )
        optimal_action, _ = controller.get_action(position, velocity, target)

        # Recalculate the noisy action until it passes the threshold
        action_magnitude = np.linalg.norm(optimal_action)
        noisy_action = np.clip(
            (optimal_action + np.random.randn(*optimal_action.shape) * noise_level),
            -1,
            1,
        )
        noisy_action = noisy_action / np.linalg.norm(noisy_action) * action_magnitude
        cosine_similarity = np.dot(optimal_action, noisy_action) / (
            np.linalg.norm(optimal_action) * np.linalg.norm(noisy_action)
        )
        cosine_distance = 1 - cosine_similarity

        while remove_optimal_actions and (cosine_distance < optimality_threshold):
            action_magnitude = np.linalg.norm(optimal_action)
            noisy_action = np.clip(
                (optimal_action + np.random.randn(*optimal_action.shape) * noise_level),
                -1,
                1,
            )
            noisy_action = (
                noisy_action / np.linalg.norm(noisy_action) * action_magnitude
            )
            cosine_similarity = np.dot(optimal_action, noisy_action) / (
                np.linalg.norm(optimal_action) * np.linalg.norm(noisy_action)
            )
            cosine_distance = 1 - cosine_similarity

        # Take a step in the environment
        next_observations, reward, terminal, info = env.step(noisy_action)
        if remove_velocity:
            next_observations = next_observations[:2]

        # Save to dataset
        new_dataset["observations"].append(dataset["observations"][i])
        new_dataset["actions"].append(noisy_action)
        new_dataset["next_observations"].append(next_observations)
        new_dataset["rewards"].append(reward)
        new_dataset["terminals"].append(terminal)

    # Optionally save to disk
    if save_path is not None:
        np.savez_compressed(save_path, **new_dataset)

    return new_dataset


def test_optimal_actions_no_velocity():
    from jaxrl_m.envs.pointmaze import PointMaze

    env = PointMaze("large")
    obs = env.reset()
    done = False
    num_steps = 0

    while not done:
        controller = waypoint_controller.WaypointController(
            env.str_maze_spec, p_gain=1.0, d_gain=0.0
        )
        action, _ = controller.get_action(obs, np.zeros_like(obs), env.get_target())
        obs, rew, done, info = env.step(action)
        num_steps += 1
        if rew == 1.0:
            break
    print("num_steps:", num_steps)


def test_velocity(env_name: str, random_seed: int = 0):

    def generator():
        env = gym.make(env_name)
        env.seed(random_seed)
        target = env.get_target()

        d_gain = -1.0
        p_gain = 10.0
        num_steps_dataset_velocity = 0
        rewards = 0

        done = False
        obs = env.reset()

        while not done:
            position = obs[:2]
            velocity = obs[2:4]

            controller = waypoint_controller.WaypointController(
                env.str_maze_spec, p_gain=p_gain, d_gain=d_gain
            )
            action, _ = controller.get_action(position, velocity, target)
            obs, rew, done, info = env.step(action)
            num_steps_dataset_velocity += 1
            rewards += rew
            # print("reward:", rew, "done:", done)
            yield

        print(
            "Number of steps with dataset velocity:",
            num_steps_dataset_velocity,
            rewards,
        )

    for _ in tqdm(generator()):
        pass

    def generator():
        env = gym.make(env_name)
        env.seed(random_seed)
        target = env.get_target()

        d_gain = -1.0
        p_gain = 10.0
        num_steps_0_velocity = 0
        rewards = 0

        done = False
        obs = env.reset()

        while not done:
            position = obs[:2]
            velocity = np.array((0, 0), dtype=np.float32)

            controller = waypoint_controller.WaypointController(
                env.str_maze_spec, p_gain=p_gain, d_gain=d_gain
            )
            action, _ = controller.get_action(position, velocity, target)

            obs, rew, done, info = env.step(action)
            num_steps_0_velocity += 1
            rewards += rew
            # print("reward:", rew, "done:", done)
            yield

        print("Number of steps with 0 velocity:", num_steps_0_velocity, rewards)

    for _ in tqdm(generator()):
        pass


def generate_dataset_downsampling_goal_radius(
    goal_radius: float,
    save_path: Optional[str],
    num_samples: int,
    additional_goal_radius_downsampling_rate: float,
    random_seed: int = 0,
    goal_location: Tuple[float, float] = (7, 9),
) -> Dict[str, np.ndarray]:
    """
    Generate a downsampled dataset for a given goal radius.
    Args:
        goal_radius (float): Goal radius.
        save_path (Optional[str]): Path to save the downsampled dataset.
        num_samples (int): Number of samples in the downsampled dataset.
        additional_goal_radius_downsampling_rate (float): Rate at which samples within the goal radius are downsampled. This is in addition to the downsampling rate of the entire dataset.
        random_seed (int): Random seed.
        goal_location (Tuple[float, float]): Tuple of (row, col) representing the location of the goal in the grid.
    Returns:
        Dict[str, np.ndarray]: Downsampled dataset.
    """
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.random.seed(random_seed)
    dataset = generate_downsampled_dataset(
        "maze2d-large-v1", num_samples, None, random_seed
    )
    samples_within_goal_radius_indices = np.where(
        np.linalg.norm(dataset["observations"][:, :2] - goal_location, axis=1)
        <= goal_radius
    )[0]
    samples_not_within_goal_radius_indices = np.where(
        np.linalg.norm(dataset["observations"][:, :2] - goal_location, axis=1)
        > goal_radius
    )[0]
    num_samples_within_goal_radius = int(
        len(samples_within_goal_radius_indices)
        * additional_goal_radius_downsampling_rate
    )
    sampled_goal_radius_indices = np.random.choice(
        samples_within_goal_radius_indices,
        num_samples_within_goal_radius,
        replace=False,
    )
    sampled_indices = np.concatenate(
        [samples_not_within_goal_radius_indices, sampled_goal_radius_indices]
    )
    sampled_dataset = {}
    for key in dataset.keys():
        sampled_dataset[key] = dataset[key][sampled_indices]

    if save_path is not None:
        np.savez_compressed(save_path, **sampled_dataset)
        # Save a plot of the dataset, with a circle indicating the goal radius, and a red dot for
        # the goal.
        plt.cla()
        x_positions = sampled_dataset["observations"][:, 0]
        y_positions = sampled_dataset["observations"][:, 1]
        plt.scatter(x_positions, y_positions, s=1)
        circle = plt.Circle(
            goal_location, goal_radius, color="r", fill=False, linewidth=1
        )
        plt.gca().add_patch(circle)
        plt.scatter(goal_location[0], goal_location[1], color="r", s=1)
        plt.savefig(save_path.replace(".npz", ".png"))

    return sampled_dataset
