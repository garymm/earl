# Earl: a framework for reinforcement learning research code

Earl is a library of reinforcement learning (RL) building blocks that strives to makes it easy to build simple, efficient, and readable agents.

The word "earl" was chosen because it's a cool word that ends in "rl" 🙂. According to [Wikipedia](https://en.wikipedia.org/wiki/Earl):

> Earl (/ɜːrl, ɜːrəl/) is a rank of the nobility in the United Kingdom.

Earl supports only JAX agents.

For pure JAX environments, Earl supports the [Gymnax](https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/environment.py) interface.
For other environments, Earl will support the [Gymnasium](https://gymnasium.farama.org/api/env/) interface.

## Development

Currently running tests with bazel is only supported on Linux x86_64.

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
