import jax
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def make_single_trajectory_visual(
    q_estimates,
    target_q_pred,
    rewards,
    masks,
    obs_images,
    goal_images,
    bellman_loss,
):
    def np_unstack(array, axis):
        arr = np.split(array, array.shape[axis], axis)
        arr = [a.squeeze() for a in arr]
        return arr

    def process_images(images):
        # assume image in C, H, W shape
        assert len(images.shape) == 4
        assert images.shape[-1] == 3

        interval = max(1, images.shape[0] // 4)

        sel_images = images[::interval]
        sel_images = np.concatenate(np_unstack(sel_images, 0), 1)
        return sel_images

    fig, axs = plt.subplots(7, 1, figsize=(8, 15))
    canvas = FigureCanvas(fig)
    plt.xlim([0, len(q_estimates)])

    obs_images = process_images(obs_images)
    goal_images = process_images(goal_images)

    axs[0].imshow(obs_images)
    axs[1].imshow(goal_images)

    # two Q functions
    axs[2].plot(q_estimates[0, :], linestyle="--", marker="o")
    axs[2].plot(q_estimates[1, :], linestyle="--", marker="o")
    axs[2].set_ylabel("q values")

    axs[3].plot(target_q_pred[0, :], linestyle="--")
    axs[3].plot(target_q_pred[1, :], linestyle="--")
    axs[3].plot(target_q_pred.min(axis=0), marker="o")
    axs[3].set_ylabel("target_q_pred")
    axs[3].set_xlim([0, len(target_q_pred)])

    axs[4].plot(bellman_loss, linestyle="--", marker="o")
    axs[4].set_ylabel("bellman_loss")
    axs[4].set_xlim([0, len(bellman_loss)])

    axs[5].plot(rewards, linestyle="--", marker="o")
    axs[5].set_ylabel("rewards")
    axs[5].set_xlim([0, len(rewards)])

    axs[6].plot(masks, linestyle="--", marker="o")
    axs[6].set_ylabel("masks")
    axs[6].set_xlim([0, len(masks)])

    plt.tight_layout()

    canvas.draw()
    out_image = np.frombuffer(canvas.buffer_rgba(), dtype="uint8")
    out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (4,))

    return out_image


def value_and_reward_visulization(trajs, agent, discount=0.99, seed=0):
    def _batch_dicts(stat):
        """stat is a list of dict, turn it into a dict of list"""
        d = {}
        for k in stat[0].keys():
            d[k] = np.array([s[k] for s in stat])
        return d

    rng = jax.random.PRNGKey(seed)
    n_trajs = len(trajs)
    visualization_images = []

    # for each trajectory
    for i in range(n_trajs):
        observations = _batch_dicts(trajs[i]["observation"])
        next_observations = _batch_dicts(trajs[i]["next_observation"])
        goals = _batch_dicts(trajs[i]["goal"])
        actions = np.array(trajs[i]["action"])
        rewards = np.array(trajs[i]["reward"])
        masks = np.array([not d for d in trajs[i]["done"]])

        q_pred = agent.forward_critic(
            (observations, goals), actions, rng=None, train=False
        )
        next_actions, _ = agent.forward_policy_and_sample(
            (next_observations, goals), rng
        )
        target_q_pred = agent.forward_critic(
            (next_observations, goals), next_actions, rng=None, train=False
        )
        td_target = rewards + target_q_pred.min(axis=0) * discount * masks
        td_loss = ((q_pred - td_target) ** 2).mean(axis=0)

        visualization_images.append(
            make_single_trajectory_visual(
                q_pred,
                target_q_pred,
                rewards,
                masks,
                observations["image"],
                goals["image"],
                td_loss,
            )
        )

    return np.concatenate(visualization_images, 0)
