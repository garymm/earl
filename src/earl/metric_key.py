import enum


class MetricKey(enum.StrEnum):
    """Keys for the metrics that will be returned by GymnaxLoop.run().

    Note that agents can return additional metrics.
    """

    ACTION_COUNTS = enum.auto()
    DURATION_SEC = enum.auto()
    LOSS = enum.auto()
    REWARD_SUM = enum.auto()
    """Sum of reward across envs."""
    REWARD_MEAN = enum.auto()
    """Mean of reward across envs."""
    TOTAL_DONES = enum.auto()
    """Sum of the number of times envs reported being done."""
    COMPLETE_EPISODE_LENGTH_MEAN = enum.auto()
    NUM_ENVS_THAT_DID_NOT_COMPLETE = enum.auto()
