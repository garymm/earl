# Earl: a framework for reinforcement learning research code

Earl is a library of reinforcement learning (RL) building blocks that strives to expose simple, efficient, and readable agents. The building blocks are designed in such a way that the agents can be run at multiple scales (e.g. single-stream vs. distributed agents).

The word "earl" was chosen because it's a cool word that ends in "rl" 🙂.

Earl supports only JAX agents.

For pure JAX environments, Earl supports the [Gymnax](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/environment.py) interface.
For other environments, Earl supports the [Gymnasium](https://gymnasium.farama.org/api/env/) interface.

## TODO

- Add checkpointing, probably use [Orbax](https://orbax.readthedocs.io/en/latest/orbax_checkpoint_101.html)
- Add support for data parallel training use [pmean](https://astralord.github.io/posts/exploring-parallel-strategies-with-jax/#data-parallelism)
- Add logging
- gymnasium loop:
  - Multi-threaded to overlap model forward, data copying, and envs step. Basically steal ideas from podracers Sebulba.
- Add builders, based on [Acme](https://github.com/google-deepmind/acme/blob/master/acme/agents/jax/builders.py).
