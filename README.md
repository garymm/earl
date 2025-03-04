# Earl: a framework for scalable reinforcement learning research

Earl is a library of reinforcement learning (RL) building blocks that strives to makes it easy to build simple, efficient, and readable agents.

Earl is built on [JAX](https://docs.jax.dev/en/latest/) and [Equinox](https://docs.kidger.site/equinox/).

Earl implements the two architectures described in ["Podracer architectures for scalable Reinforcement Learning"](https://arxiv.org/abs/2104.06272), which were used at DeepMind to scale training to very large batch sizes across many chips. This repository includes a few agents (AKA RL algorithms), notably R2D2 as described in "Recurrent Experience Replay In Distributed Reinforcement Learning".

The most important parts of Earl are:

* The [Agent](earl/core.py) abstract base class. It is designed to be flexible enough to allow implementation of a wide variety of RL algorithm, but structured enough to allow for standardized environment loops to be used for training all such agents. Earl agents are implemented using Equinox.
* [GymnaxLoop](earl/environment_loop/gymnax_loop.py). For jax.jit-compatible environments, Earl supports the [Gymnax](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/environment.py) interface. GymnaxLoop implements distributed training for these environments. This implements the Anakin architecture from the Podracer paper.
* [GymnasiumLoop](earl/environment_loop/gymnasium_loop.py). For other environments, Earl will support the [Gymnasium](https://gymnasium.farama.org/api/env/) interface. GymnasiumLoop implements distributed training for these environments. This implements the Sebulba architecture from the Podracer paper.

For an example of training an agent using both loops, see this [notebook](earl/agents/r2d2/train_r2d2_asterix.ipynb).

Included example agent implemented in Earl:

* [Simple Policy Gradient](earl/agents/simple_policy_gradient/simple_policy_gradient.py) (very simple).
* [R2D2](earl/agents/r2d2/r2d2.py) (quite complicated).


There is currently no package on PyPi, but Earl is pure Python, so it can be installed from source, e.g.:

```sh
uv pip install "earl @ git+https://github.com/garymm/earl.git"
```

[Here's a blog post](https://www.garymm.org/blog/2025/03/03/earl/) that discusses some of the rationale and lessons learned developing Earl.

## Development

### Testing

[![codecov](https://codecov.io/gh/garymm/earl/graph/badge.svg?token=MDG3TCNML8)](https://codecov.io/gh/garymm/earl)


There are two ways to run tests:


#### Bazel

Testing with Bazel is what happens in GitHub workflows. It's good for running lots of tests in parallel and it intelligently caches results.
However it has a high overhead so running an individual test is slower and it doesn't have the same level of control as Pytest.

Currently running tests with Bazel is only supported on Linux x86_64.

Install [Bazelisk](https://github.com/bazelbuild/bazelisk/blob/master/README.md), name it `bazel`.

Then run tests with:

```sh
bazel test //...
```


#### Pytest

To set this up first create a virtual environment (see section below), then `source .venv/bin/activate`.
Then you can run `pytest` normally.

### VEnv / IDE Setup

When using VS Code intall the recommended extensions by searching for `@recommended`.

Set up a virtual environment with:

```sh
bazel run //:dot_venv_linux_x86_64
```

This will create a `.venv` directory with the dependencies so you can use it with your IDE.

## Citation

If you use Earl in your research, please cite it:

```bibtex
@software{miguel2024earl,
  author = {Miguel, Gary},
  title = {Earl: A Framework for Scalable Reinforcement Learning Research},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/garymm/earl},
  description = {A library of reinforcement learning (RL) building blocks that strives to makes it easy to build simple, efficient, and readable agents}
}
```
