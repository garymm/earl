import copy
from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping


class ResidualBlock(eqx.Module):
  """Residual block of operations, e.g. convolutional."""

  inner_op1: eqx.nn.Conv2d
  inner_op2: eqx.nn.Conv2d
  layernorm1: eqx.nn.LayerNorm | None
  layernorm2: eqx.nn.LayerNorm | None
  use_layer_norm: bool

  def __init__(
    self,
    num_channels: int,
    use_layer_norm: bool = False,
    dtype: jnp.dtype = jnp.float32,
    *,
    key: jaxtyping.PRNGKeyArray,
  ):
    keys = jax.random.split(key, 2)
    self.inner_op1 = eqx.nn.Conv2d(
      num_channels, num_channels, kernel_size=3, padding=1, key=keys[0], dtype=dtype
    )
    self.inner_op2 = eqx.nn.Conv2d(
      num_channels, num_channels, kernel_size=3, padding=1, key=keys[1], dtype=dtype
    )
    self.use_layer_norm = use_layer_norm

    if use_layer_norm:
      # Shape is (channels) for the normalization, we'll vmap over H,W
      self.layernorm1 = eqx.nn.LayerNorm(num_channels, eps=1e-6, dtype=dtype)
      self.layernorm2 = eqx.nn.LayerNorm(num_channels, eps=1e-6, dtype=dtype)
    else:
      self.layernorm1 = None
      self.layernorm2 = None

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    output = x

    # First layer in residual block
    if self.layernorm1 is not None:
      # Transpose to (H, W, C) for LayerNorm
      output = jnp.transpose(output, (1, 2, 0))
      # Apply LayerNorm to channel dimension at each spatial location
      output = jax.vmap(jax.vmap(self.layernorm1))(output)
      # Transpose back to (C, H, W)
      output = jnp.transpose(output, (2, 0, 1))

    output = jax.nn.relu(output)
    output = self.inner_op1(output)

    # Second layer in residual block
    if self.layernorm2 is not None:
      # Same normalization pattern as above
      output = jnp.transpose(output, (1, 2, 0))
      output = jax.vmap(jax.vmap(self.layernorm2))(output)
      output = jnp.transpose(output, (2, 0, 1))

    output = jax.nn.relu(output)
    output = self.inner_op2(output)
    return x + output


def make_downsampling_layer(
  in_channels: int, out_channels: int, *, key: jaxtyping.PRNGKeyArray, dtype: jnp.dtype
) -> tuple[eqx.nn.Conv2d, eqx.nn.MaxPool2d]:
  """Returns conv + maxpool layers for downsampling."""
  conv = eqx.nn.Conv2d(
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=3,
    stride=1,
    padding=1,
    key=key,
    dtype=dtype,
  )
  maxpool = eqx.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
  return conv, maxpool


class ResNetTorso(eqx.Module):
  """ResNetTorso for visual inputs, inspired by the IMPALA paper."""

  in_channels: int = eqx.field(static=True)
  channels_per_group: Sequence[int] = eqx.field(static=True)
  blocks_per_group: Sequence[int] = eqx.field(static=True)
  use_layer_norm: bool = eqx.field(static=True)
  downsampling_layers: list[tuple[eqx.nn.Conv2d, eqx.nn.MaxPool2d]]
  residual_blocks: list[list[ResidualBlock]]

  def __init__(
    self,
    in_channels: int = 4,  # 4 stacked frames
    channels_per_group: Sequence[int] = (16, 32, 32),
    blocks_per_group: Sequence[int] = (2, 2, 2),
    use_layer_norm: bool = False,
    dtype: jnp.dtype = jnp.float32,
    *,
    key: jaxtyping.PRNGKeyArray,
  ):
    self.in_channels = in_channels
    self.channels_per_group = channels_per_group
    self.blocks_per_group = blocks_per_group
    self.use_layer_norm = use_layer_norm

    if len(channels_per_group) != len(blocks_per_group):
      raise ValueError(
        "Length of channels_per_group and blocks_per_group must be equal. "
        f"Got channels_per_group={channels_per_group}, "
        f"blocks_per_group={blocks_per_group}"
      )

    # Create keys for all layers
    num_groups = len(channels_per_group)
    total_blocks = sum(blocks_per_group)
    keys = jax.random.split(key, num_groups + total_blocks)

    # Create downsampling layers
    downsample_keys = keys[:num_groups]
    prev_channels = in_channels
    self.downsampling_layers = []
    for channels, k in zip(channels_per_group, downsample_keys, strict=False):
      layer = make_downsampling_layer(
        in_channels=prev_channels, out_channels=channels, key=k, dtype=dtype
      )
      self.downsampling_layers.append(layer)
      prev_channels = channels

    # Create residual blocks
    block_keys = keys[num_groups:]
    key_idx = 0
    self.residual_blocks = []
    for channels, num_blocks in zip(channels_per_group, blocks_per_group, strict=False):
      group_blocks = []
      for _ in range(num_blocks):
        block = ResidualBlock(
          num_channels=channels, use_layer_norm=use_layer_norm, key=block_keys[key_idx]
        )
        group_blocks.append(block)
        key_idx += 1
      self.residual_blocks.append(group_blocks)

  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    output = inputs

    for (conv, maxpool), blocks in zip(
      self.downsampling_layers, self.residual_blocks, strict=False
    ):
      # Downsampling
      output = conv(output)
      output = maxpool(output)

      # Residual blocks
      for block in blocks:
        output = block(output)

    return output


class DeepAtariTorso(eqx.Module):
  """Deep torso for Atari, from the IMPALA paper.

  Based on
  https://github.com/google-deepmind/acme/blob/eedf63ca039856876ff85be472fa9186cf29b073/acme/jax/networks/atari.py
  """

  resnet: ResNetTorso
  mlp_head: eqx.nn.MLP
  use_layer_norm: bool = eqx.field(static=True)

  def __init__(
    self,
    channels_per_group: Sequence[int] = (16, 32, 32),
    blocks_per_group: Sequence[int] = (2, 2, 2),
    hidden_sizes: Sequence[int] = (512,),
    use_layer_norm: bool = True,
    in_channels: int = 4,
    input_size: int = 84,
    dtype: jnp.dtype = jnp.float32,
    *,
    key: jaxtyping.PRNGKeyArray,
  ):
    keys = jax.random.split(key, 2)
    self.use_layer_norm = use_layer_norm
    self.resnet = ResNetTorso(
      channels_per_group=channels_per_group,
      blocks_per_group=blocks_per_group,
      use_layer_norm=use_layer_norm,
      in_channels=in_channels,
      dtype=dtype,
      key=keys[0],
    )

    # Calculate spatial size through actual pooling steps
    spatial_size = input_size
    for _ in range(len(channels_per_group)):
      spatial_size = (spatial_size + 2 * 1 - 3) // 2 + 1  # MaxPool formula

    spatial_dim = spatial_size * spatial_size
    in_size = channels_per_group[-1] * spatial_dim

    self.mlp_head = eqx.nn.MLP(
      in_size=in_size,
      out_size=hidden_sizes[-1],
      width_size=hidden_sizes[0],
      depth=len(hidden_sizes),
      activation=jax.nn.relu,
      final_activation=jax.nn.relu,
      dtype=dtype,
      key=keys[1],
    )

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    if x.dtype == jnp.uint8:
      x = x.astype(jnp.float32) / 255.0
    output = self.resnet(x)
    output = jax.nn.relu(output)
    # Flatten all dimensions into a single vector
    output = output.reshape(-1)
    output = self.mlp_head(output)
    return output


class OAREmbedding(eqx.Module):
  """Module for embedding (observation, action, reward) inputs together.

  Based on
  https://github.com/google-deepmind/acme/blob/eedf63ca039856876ff85be472fa9186cf29b073/acme/jax/networks/embedding.py
  """

  torso: Callable[[jax.Array], jax.Array]
  num_actions: int

  def __call__(self, observation: jax.Array, action: jax.Array, reward: jax.Array) -> jnp.ndarray:
    """Embed each of the (observation, action, reward) inputs & concatenate."""
    features = self.torso(observation)  # [D]

    action = jnp.squeeze(jax.nn.one_hot(action, num_classes=self.num_actions))  # [A]
    # Map rewards -> [-1, 1].
    reward = jnp.tanh(reward)
    # Add dummy trailing dimensions to rewards if necessary.
    while reward.ndim < action.ndim:
      reward = jnp.expand_dims(reward, axis=-1)

    # Concatenate on final dimension.
    embedding = jnp.concatenate([features, action, reward], axis=-1)  # [D+A+1]
    return embedding


class R2D2Network(eqx.Module):
  """The R2D2 network: a convolutional feature extractor, an LSTMCell, and a dueling head."""

  embed: Callable[
    [jax.Array, jax.Array, jax.Array], jax.Array
  ]  # Section 2.3: convolutional feature extractor.
  lstm_cell: eqx.nn.LSTMCell  # Section 2.3 & 3: recurrent cell.
  dueling_value: eqx.nn.Linear  # Section 2.3: value branch.
  dueling_advantage: eqx.nn.Linear  # Section 2.3: advantage branch.

  def __init__(
    self,
    torso: Callable[[jax.Array], jax.Array],
    lstm_cell: eqx.nn.LSTMCell,
    dueling_value: eqx.nn.Linear,
    dueling_advantage: eqx.nn.Linear,
    num_actions: int,
  ):
    self.embed = OAREmbedding(torso=torso, num_actions=num_actions)
    self.lstm_cell = lstm_cell
    self.dueling_value = dueling_value
    self.dueling_advantage = dueling_advantage

  def __call__(
    self,
    observation: jax.Array,
    action: jax.Array,
    reward: jax.Array,
    hidden: tuple[jax.Array, jax.Array],
  ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    features = self.embed(observation, action, reward)
    h, c = self.lstm_cell(features, hidden)
    value = self.dueling_value(h)
    advantage = self.dueling_advantage(h)
    q_values = value + (advantage - jnp.mean(advantage, axis=-1, keepdims=True))
    return q_values, (h, c)


class R2D2Networks(eqx.Module):
  online: R2D2Network
  target: R2D2Network


def make_networks_resnet(
  num_actions: int,
  in_channels: int,
  dtype: jnp.dtype = jnp.float32,
  hidden_size: int = 512,
  *,
  key: jaxtyping.PRNGKeyArray,
) -> R2D2Networks:
  torso_key, lstm_key, dueling_value_key, dueling_advantage_key = jax.random.split(key, 4)

  online = R2D2Network(
    torso=DeepAtariTorso(
      in_channels=in_channels,
      # output will be concatenated with action and reward, so we subtract them from the hidden size
      hidden_sizes=(hidden_size - num_actions - 1,),
      dtype=dtype,
      key=torso_key,
    ),
    lstm_cell=eqx.nn.LSTMCell(hidden_size, hidden_size, dtype=dtype, key=lstm_key),
    dueling_value=eqx.nn.Linear(hidden_size, 1, dtype=dtype, key=dueling_value_key),
    dueling_advantage=eqx.nn.Linear(
      hidden_size, num_actions, dtype=dtype, key=dueling_advantage_key
    ),
    num_actions=num_actions,
  )
  target = copy.deepcopy(online)
  return R2D2Networks(online=online, target=target)
