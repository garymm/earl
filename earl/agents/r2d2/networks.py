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
    *,
    key: jaxtyping.PRNGKeyArray,
  ):
    keys = jax.random.split(key, 2)
    self.inner_op1 = eqx.nn.Conv2d(
      num_channels, num_channels, kernel_size=3, padding=1, key=keys[0]
    )
    self.inner_op2 = eqx.nn.Conv2d(
      num_channels, num_channels, kernel_size=3, padding=1, key=keys[1]
    )
    self.use_layer_norm = use_layer_norm

    if use_layer_norm:
      # Shape is (channels, height, width) for the normalization
      self.layernorm1 = eqx.nn.LayerNorm((num_channels, 1, 1), eps=1e-6)
      self.layernorm2 = eqx.nn.LayerNorm((num_channels, 1, 1), eps=1e-6)
    else:
      self.layernorm1 = None
      self.layernorm2 = None

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    output = x

    # First layer in residual block
    if self.layernorm1 is not None:
      output = self.layernorm1(output)
    output = jax.nn.relu(output)
    output = self.inner_op1(output)

    # Second layer in residual block
    if self.layernorm2 is not None:
      output = self.layernorm2(output)
    output = jax.nn.relu(output)
    output = self.inner_op2(output)
    return x + output


def make_downsampling_layer(
  output_channels: int, *, key: jaxtyping.PRNGKeyArray
) -> tuple[eqx.nn.Conv2d, eqx.nn.MaxPool2d]:
  """Returns conv + maxpool layers for downsampling."""
  conv = eqx.nn.Conv2d(
    in_channels=output_channels,
    out_channels=output_channels,
    kernel_size=3,
    stride=1,
    padding=1,
    key=key,
  )
  maxpool = eqx.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
  return conv, maxpool


class ResNetTorso(eqx.Module):
  """ResNetTorso for visual inputs, inspired by the IMPALA paper."""

  channels_per_group: Sequence[int]
  blocks_per_group: Sequence[int]
  use_layer_norm: bool
  downsampling_layers: list[tuple[eqx.nn.Conv2d, eqx.nn.MaxPool2d]]
  residual_blocks: list[list[ResidualBlock]]

  def __init__(
    self,
    channels_per_group: Sequence[int] = (16, 32, 32),
    blocks_per_group: Sequence[int] = (2, 2, 2),
    use_layer_norm: bool = False,
    *,
    key: jaxtyping.PRNGKeyArray,
  ):
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
    self.downsampling_layers = [
      make_downsampling_layer(channels, key=k)
      for channels, k in zip(channels_per_group, downsample_keys, strict=False)
    ]

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
  use_layer_norm: bool

  def __init__(
    self,
    channels_per_group: Sequence[int] = (16, 32, 32),
    blocks_per_group: Sequence[int] = (2, 2, 2),
    hidden_sizes: Sequence[int] = (512,),
    use_layer_norm: bool = True,
    *,
    key: jaxtyping.PRNGKeyArray,
  ):
    keys = jax.random.split(key, 2)
    self.use_layer_norm = use_layer_norm
    self.resnet = ResNetTorso(
      channels_per_group=channels_per_group,
      blocks_per_group=blocks_per_group,
      use_layer_norm=use_layer_norm,
      key=keys[0],
    )

    # MLP head with activation on final layer
    self.mlp_head = eqx.nn.MLP(
      in_size=channels_per_group[-1],  # Last channel count
      out_size=hidden_sizes[-1],
      width_size=hidden_sizes[0] if len(hidden_sizes) > 1 else hidden_sizes[0],
      depth=len(hidden_sizes),
      activation=jax.nn.relu,
      final_activation=jax.nn.relu,
      key=keys[1],
    )

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    output = self.resnet(x)
    output = jax.nn.relu(output)
    # Flatten all dimensions except batch and channel
    output = output.reshape(output.shape[0], -1)
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
    action = jax.nn.one_hot(action, num_classes=self.num_actions)  # [A]
    # Map rewards -> [-1, 1].
    reward = jnp.tanh(reward)
    # Add dummy trailing dimensions to rewards if necessary.
    while reward.ndim < action.ndim:
      reward = jnp.expand_dims(reward, axis=-1)
    # Concatenate on final dimension.
    embedding = jnp.concatenate([features, action, reward], axis=-1)  # [D+A+1]
    return embedding
