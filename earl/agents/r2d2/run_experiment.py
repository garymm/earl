import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random

import earl.agents.r2d2.networks
import earl.agents.r2d2.r2d2 as r2d2

key = jax.random.PRNGKey(0)
torso_key, lstm_key, dueling_value_key, dueling_advantage_key = jax.random.split(key, 4)
network = r2d2.R2D2Network(
  torso=earl.agents.r2d2.networks.DeepAtariTorso(
    key=torso_key,
  ),
  lstm_cell=eqx.nn.LSTMCell(512, 512, key=lstm_key),
  dueling_value=eqx.nn.Linear(512, 1, key=dueling_value_key),
  dueling_advantage=eqx.nn.Linear(512, 1, key=dueling_advantage_key),
  num_actions=18,
)

obs = jax.random.randint(key, (1, 84, 84, 4), 0, 256)
action = jax.random.randint(key, (1,), 0, 18)
reward = jax.random.uniform(key, (1,))
hidden = (jnp.zeros((1, 512)), jnp.zeros((1, 512)))

print(network(obs, action, reward, hidden))
