import enum


class MetricKey(enum.StrEnum):
  """Keys for the metrics that will be returned by GymnaxLoop.run().

  Note that agents can return additional metrics.
  """

  ACTION_COUNTS = enum.auto()
  DURATION_SEC = enum.auto()
  ACTOR_WAIT_DURATION_SEC = enum.auto()
  """Time spent waiting for actor to complete.

  Note with multiple devices this is the time that is NOT hidden by
  overlapping acting and learning.
  """
  LEARN_DURATION_SEC = enum.auto()
  """Time spent waiting for learning to complete."""
  LOSS = enum.auto()
  REWARD_SUM = enum.auto()
  """Sum of reward across envs."""
  REWARD_MEAN = enum.auto()
  """Mean of reward across envs."""
  TOTAL_DONES = enum.auto()
  """Sum of the number of times envs reported being done."""
  COMPLETE_EPISODE_LENGTH_MEAN = enum.auto()
  NUM_ENVS_THAT_DID_NOT_COMPLETE = enum.auto()
