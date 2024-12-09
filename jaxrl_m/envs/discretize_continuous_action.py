import jax
import numpy as np

ACTION_DIM = None
N_BINS_PER_ACTION = None
ACTION_BINS = None
GAMMA = None


def onehot_actions(action_index, n_actions=None):
    """turn a batch of actions of size (batch_size, ) into onehot actions"""
    if n_actions is None:
        n_actions = N_BINS_PER_ACTION
    batch_size = len(action_index)
    onehot = np.zeros((batch_size, n_actions))
    onehot[np.arange(batch_size), action_index] = 1
    return onehot


def batch_onehot_actions(onehots, batch_size):
    """
    Assume that `onehots` is already onehot actions, and adjacent actions are adjacent Q-transformer-step actions
    We pad 0's at the end of enable action history, for appending to the state
    Args:
        onehots: (batch_size * action_dim, n_bins_per_action)
    returns:
        (batch_size * action_dim, n_bins_per_action * action_dim)
    """
    assert onehots.shape[0] == batch_size * ACTION_DIM
    assert onehots.shape[1] == N_BINS_PER_ACTION

    onehots_history = np.zeros(
        (batch_size * ACTION_DIM, N_BINS_PER_ACTION * ACTION_DIM)
    )
    for b in range(batch_size):
        for i in range(ACTION_DIM):
            onehots_history[b * ACTION_DIM + i, 0 : i * N_BINS_PER_ACTION] = onehots[
                b * ACTION_DIM : b * ACTION_DIM + i, :
            ].flatten()
    return onehots_history


def q_transformer_action_discretization_init(
    action_space, num_bins_per_action_dim, discount
):
    """init the global vars in this file
    this function should be called before any other q_transformer functions
    """
    global ACTION_DIM, ACTION_BINS, GAMMA, N_BINS_PER_ACTION
    ACTION_DIM = action_space.shape[0]
    low = action_space.low
    high = action_space.high
    ACTION_BINS = [
        np.linspace(low[i], high[i], num_bins_per_action_dim[i])
        for i in range(ACTION_DIM)
    ]
    GAMMA = discount

    assert np.all(
        num_bins_per_action_dim == num_bins_per_action_dim[0]
    ), "currently only support same number of bins per action dim"
    N_BINS_PER_ACTION = num_bins_per_action_dim[0]


def q_transformer_action_discretization(batch):
    """Discretize the continuous action space Q-transformer style
    Discretize the actions along each dimension, and the Q-function only predicts the discretized bin.
    The action should also just be one-hot actions
    The rewards along the way should also be changed to only apply at the last step
    The state should be augmented to include the dim-actions taken so far at this env step
    args:
        batch: a dictionary of s, a, r, s', mask
    returns:
        the same dictionary, but the batchsize is multiplied by dim_actions
    """
    actions = batch["actions"]
    batch_size = batch["actions"].shape[0]

    # discretize actions by dimension
    discretized_actions_per_dim = []  # len action_dim, each elt is (1, batch_size)
    for i in range(ACTION_DIM):
        a = (
            np.digitize(actions[:, i], ACTION_BINS[i]).reshape(1, -1) - 1
        )  # (1, batch_size), -1 for 0-indexing actions
        discretized_actions_per_dim.append(a)
    # reshape so that each OG batch is together
    new_actions = np.vstack(
        discretized_actions_per_dim
    ).T.flatten()  # (batch_size * action_dim, )

    # append the onehot actions to the state
    new_observations = np.zeros(
        (
            batch_size * ACTION_DIM,
            batch["observations"].shape[1] + ACTION_DIM * N_BINS_PER_ACTION,
        )
    )
    new_observations[:, 0 : batch["observations"].shape[1]] = np.repeat(
        batch["observations"], ACTION_DIM, axis=0
    )
    new_observations[:, batch["observations"].shape[1] :] = batch_onehot_actions(
        onehot_actions(new_actions), batch_size
    )

    new_next_observations = np.zeros_like(new_observations)
    # shift forward by 1
    new_next_observations[0:-1, :] = new_observations[1:, :]
    # sub in the next_observation at the last action-dim-step
    new_next_observations[
        ACTION_DIM - 1 :: ACTION_DIM, 0 : batch["observations"].shape[1]
    ] = batch["next_observations"]

    # apply the original reward, mask, discount at the 0-th action-dim-step
    new_rewards = np.zeros((batch_size * ACTION_DIM,))
    new_rewards[ACTION_DIM - 1 :: ACTION_DIM] = batch["rewards"]

    new_mask = np.ones_like(new_rewards)
    new_mask[ACTION_DIM - 1 :: ACTION_DIM] = batch["masks"]

    new_discounts = np.ones_like(new_rewards)
    new_discounts[ACTION_DIM - 1 :: ACTION_DIM] = GAMMA

    return {
        "observations": new_observations,
        "actions": new_actions,
        "rewards": new_rewards,
        "next_observations": new_next_observations,
        "masks": new_mask,
        "discounts": new_discounts,
    }


def q_transformer_choose_actions(observations, agent, rng, temperature=1, argmax=False):
    """
    Choose actions by auto-regressively choosing each dimension of the action,
    and then recombining to get the continuous action
    """
    if observations.ndim == 1:
        # add batch dim
        observations = observations.reshape(1, -1)

    batch_size = observations.shape[0]
    og_obs_dim = observations.shape[1]

    discrete_actions = np.zeros((batch_size, ACTION_DIM), dtype=int)
    # init obs
    obs = np.zeros((batch_size, og_obs_dim + ACTION_DIM * N_BINS_PER_ACTION))
    obs[:, 0:og_obs_dim] = observations

    # auto-regressively feed in actions
    for i in range(ACTION_DIM):
        dim_a = agent.sample_actions(
            obs, seed=rng, temperature=temperature, argmax=argmax
        )  # (batch_size, 1)
        discrete_actions[:, i] = dim_a
        obs[
            :,
            og_obs_dim
            + i * N_BINS_PER_ACTION : og_obs_dim
            + (i + 1) * N_BINS_PER_ACTION,
        ] = onehot_actions(dim_a)

    if argmax:
        continuous_actions = [
            np.mean(
                [
                    np.take(ACTION_BINS[i], discrete_actions[:, i]),
                    np.take(
                        ACTION_BINS[i],
                        np.clip(
                            discrete_actions[:, i] + 1,
                            a_min=0,
                            a_max=N_BINS_PER_ACTION - 1,
                        ),
                    ),
                ]
            )
            for i in range(ACTION_DIM)
        ]
    else:
        continuous_actions = [
            np.random.uniform(
                np.take(ACTION_BINS[i], discrete_actions[:, i]),
                np.take(
                    ACTION_BINS[i],
                    np.clip(
                        discrete_actions[:, i] + 1, a_min=0, a_max=N_BINS_PER_ACTION - 1
                    ),
                ),
            )
            for i in range(ACTION_DIM)
        ]

    recombined_actions = np.hstack(continuous_actions).reshape(
        -1, ACTION_DIM
    )  # (batch_size, action_dim)

    if batch_size == 1:
        # unbatch
        recombined_actions = recombined_actions.reshape(-1)

    return recombined_actions
