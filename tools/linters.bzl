"""Based on https://github.com/aspect-build/rules_lint/blob/v1.0.9/example/tools/lint/linters.bzl"""

load("@aspect_rules_lint//lint:ruff.bzl", "lint_ruff_aspect")

ruff = lint_ruff_aspect(
    binary = "@aspect_rules_lint//format:ruff",
    configs = ["@@//:pyproject.toml"],
)
