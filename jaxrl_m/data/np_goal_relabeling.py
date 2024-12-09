import numpy as np


def uniform(traj, *, reached_proportion, discount, label_mc_returns=True):
    """
    Relabels with a true uniform distribution over future states. With
    probability reached_proportion, observations[i] gets a goal
    equal to next_observations[i].  In this case, the reward is 0. Otherwise,
    observations[i] gets a goal sampled uniformly from the set
    next_observations[i + 1:], and the reward is -1.
    """
    traj_len = np.shape(traj["terminals"])[0]

    # select a random future index for each transition i in the range [i + 1, traj_len)
    rand = np.random.uniform(size=traj_len)
    low = np.arange(traj_len) + 1
    high = traj_len
    goal_idxs = np.floor(rand * (high - low) + low).astype(np.int32)

    # sometimes there are floating-point errors that cause an out-of-bounds
    goal_idxs = np.minimum(goal_idxs, traj_len - 1)

    # select a random proportion of transitions to relabel with the next observation
    goal_reached_mask = np.random.uniform(size=traj_len) < reached_proportion

    # the last transition must be goal-reaching
    goal_reached_mask = np.logical_or(
        goal_reached_mask, np.arange(traj_len) == traj_len - 1
    )

    # make goal-reaching transitions have an offset of 0
    goal_idxs = np.where(goal_reached_mask, np.arange(traj_len), goal_idxs)

    # select goals
    traj["goals"] = np.take(traj["next_observations"], goal_idxs, axis=0)

    # reward is 0 for goal-reaching transitions, -1 otherwise
    traj["rewards"] = np.where(goal_reached_mask, 0, -1).astype(np.int32)

    # add masks
    traj["masks"] = np.logical_not(goal_reached_mask)

    # add distances to goal
    traj["goal_dists"] = goal_idxs - np.arange(traj_len)

    # add the mc returns
    if label_mc_returns:
        traj["mc_returns"] = (
            -1
            * (np.power(discount, traj["goal_dists"].astype(np.float32)) - 1)
            / (discount - 1)
        )

    return traj


def geometric(
    traj,
    *,
    reached_proportion,
    discount,
    label_mc_returns=True,
    last_state_proportion=0.0
):
    """
    Relabels with a geometric distribution over future states. With
    probability reached_proportion, observations[i] gets a goal
    equal to next_observations[i].  In this case, the reward is 0. Otherwise,
    observations[i] gets a goal sampled geometrically from the set
    next_observations[i + 1:], and the reward is -1.
    """
    traj_len = np.shape(traj["terminals"])[0]

    # geometrically select a future index for each transition i in the range [i + 1, traj_len)
    arange = np.arange(traj_len)
    is_future_mask = (arange[:, None] < arange[None]).astype(np.float32)
    d = discount ** (arange[None] - arange[:, None]).astype(np.float32)

    probs = is_future_mask * d
    # hack: last row is all 0s, and will cause division issues.
    # This is masked out by goal_reached_mask so it doesn't matter
    probs[-1, -1] = 1.0
    probs = probs / probs.sum(axis=1, keepdims=True)  # normalize
    goal_idxs = np.array(
        [
            np.random.choice(np.arange(traj_len), size=1, p=probs[i] / probs[i].sum())
            for i in range(traj_len)
        ],
        dtype=np.int32,
    )[:, 0]

    # select a random proportion of transitions to relabel with the next observation
    goal_reached_mask = np.random.uniform(size=traj_len) < reached_proportion

    # the last transition must be goal-reaching
    goal_reached_mask = np.logical_or(
        goal_reached_mask, np.arange(traj_len) == traj_len - 1
    )

    # select a random proportion of transitions to relabel with the last observation
    last_state_mask = np.random.uniform(size=traj_len) < last_state_proportion

    # set goals to the last state for the selected proportion
    goal_idxs = np.where(last_state_mask, traj_len - 1, goal_idxs)

    # make goal-reaching transitions have an offset of 0
    goal_idxs = np.where(goal_reached_mask, np.arange(traj_len), goal_idxs)

    # select goals
    traj["goals"] = np.take(traj["next_observations"], goal_idxs, axis=0)

    # reward is 0 for goal-reaching transitions, -1 otherwise
    traj["rewards"] = np.where(goal_reached_mask, 0, -1).astype(np.int32)

    # add masks
    traj["masks"] = np.logical_not(goal_reached_mask)

    # add distances to goal
    traj["goal_dists"] = goal_idxs - np.arange(traj_len)

    # add the mc return
    if label_mc_returns:
        traj["mc_returns"] = (
            -1
            * (np.power(discount, traj["goal_dists"].astype(np.float32)) - 1)
            / (discount - 1)
        )

    return traj


def last_state_upweighted(traj, *, reached_proportion, discount, label_mc_returns=True):
    """
    A weird relabeling scheme where the last state gets upweighted. For each
    transition i, a uniform random number is generated in the range [i + 1, i +
    traj_len). It then gets clipped to be less than traj_len. Therefore, the
    first transition (i = 0) gets a goal sampled uniformly from the future, but
    for i > 0 the last state gets more and more upweighted.
    """
    traj_len = np.shape(traj["terminals"])[0]

    # select a random future index for each transition
    offsets = np.random.randint(
        low=1,
        high=traj_len,
        size=traj_len,
    )

    # select random transitions to relabel as goal-reaching
    goal_reached_mask = np.random.uniform(size=traj_len) < reached_proportion
    # last transition is always goal-reaching
    goal_reached_mask = np.logical_or(
        goal_reached_mask, np.arange(traj_len) == traj_len - 1
    )

    # the goal will come from the current transition if the goal was reached
    offsets = np.where(goal_reached_mask, 0, offsets)

    # convert from relative to absolute indices
    goal_idxs = np.arange(traj_len) + offsets

    # clamp out of bounds indices to the last transition
    goal_idxs = np.clip(goal_idxs, a_min=0, a_max=traj_len - 1)

    # select goals
    traj["goals"] = np.take(traj["next_observations"], goal_idxs, axis=0)

    # reward is 0 for goal-reaching transitions, -1 otherwise
    traj["rewards"] = np.where(goal_reached_mask, 0, -1).astype(np.int32)

    # add masks
    traj["masks"] = np.logical_not(goal_reached_mask)

    # add distances to goal
    traj["goal_dists"] = goal_idxs - np.arange(traj_len)

    # add the mc return
    if label_mc_returns:
        traj["mc_returns"] = (
            -1
            * (np.power(discount, traj["goal_dists"].astype(np.float32)) - 1)
            / (discount - 1)
        )

    return traj


def specific_goal(traj, *, goal, label_mc_returns_fn=None):
    """
    Relabels with a specific goal. The goal is one specific goal in the same space
    as the observations. We label the MC returns according to the reward in the traj

    This is used in HER where we relabel with the commanded goal
    """
    traj_len = np.shape(traj["terminals"])[0]

    # label the goal
    traj["goals"] = np.tile(goal, (traj_len))

    # don't change the reward or masks
    # just add the mc returns
    if label_mc_returns_fn is not None:
        mc_returns = label_mc_returns_fn(traj["rewards"], traj["masks"])
        traj["mc_returns"] = mc_returns

    return traj


GOAL_RELABELING_FUNCTIONS = {
    "geometric": geometric,
    "uniform": uniform,
    "last_state_upweighted": last_state_upweighted,
}
