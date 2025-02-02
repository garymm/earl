# Earl: a framework for reinforcement learning research code

Earl is a library of reinforcement learning (RL) building blocks that strives to makes it easy to build simple, efficient, and readable agents.

The word "earl" was chosen because it's a cool word that ends in "rl" 🙂. According to [Wikipedia](https://en.wikipedia.org/wiki/Earl):

> Earl (/ɜːrl, ɜːrəl/) is a rank of the nobility in the United Kingdom.

Earl supports only JAX agents.

For pure JAX environments, Earl supports the [Gymnax](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/environment.py) interface.
For other environments, Earl will support the [Gymnasium](https://gymnasium.farama.org/api/env/) interface.

## Development

Install [Bazelisk](https://github.com/bazelbuild/bazelisk/blob/master/README.md), name it `bazel`.

Then run tests with:

```sh
bazel test //...
```

### IDE Setup

When using VS Code intall the recommended extensions by searching for `@recommended`.

Set up a virtual environment with:

```sh
bazel run //:dot_venv_linux_x86_64
```

This will create a `.venv` directory with the dependencies so you can use it with your IDE.



## TODO

- Add support for data parallel training use [pmean](https://astralord.github.io/posts/exploring-parallel-strategies-with-jax/#data-parallelism)
- Add support for saving evaluation observations to a video file.
- Add gymnasium loop:
  - Multi-threaded to overlap model forward, data copying, and envs step. Basically steal ideas from podracers Sebulba.
- Add builders, based on [Acme](https://github.com/google-deepmind/acme/blob/master/acme/agents/jax/builders.py).
