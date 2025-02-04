"""Wrapper macros for aspect_rules_py py_test and py_pytest_main."""

load("@aspect_rules_py//py:defs.bzl", "py_pytest_main", aspect_py_test = "py_test")

def py_test(name, srcs, main = None, deps = [], filterwarnings = [], **kwargs):
    """A macro that wraps the py_test rule from aspect_rules_py.

    This macro sets a default main attribute that points to the pytest_main target if not provided.

    Args:
      name: A string representing the test name.
      srcs: A list of source file names.
      deps: A list of dependencies.
      main: An optional string for the main target. If not provided, a default pytest_main target is generated.
      filterwarnings: A list of warnings to filter. See https://docs.python.org/3/using/cmdline.html#cmdoption-W
      **kwargs: Additional keyword arguments passed to the underlying py_test rule.

    Usage:
      load("//tools/py_test:py_test.bzl", "py_test")
      py_test(
          name = "my_test",
          srcs = ["my_test.py"],
          deps = [...],
      )
    """
    if main == None:
        main_name = "__" + name + "_pytest_main__"
        filterwarnings = ["error"] + filterwarnings
        py_pytest_main(
            name = main_name,
            deps = [
                "@pypi//coverage",
                "@pypi//pytest",
            ],
            args = [item for warning in filterwarnings for item in ["-W", warning]],
        )
        main = ":" + main_name + ".py"
        srcs.append(":" + main_name)
        deps.append(":" + main_name)
    aspect_py_test(
        name = name,
        srcs = srcs,
        main = main,
        deps = deps,
        **kwargs
    )
