#!/usr/bin/env bash
#
# based on:
# https://github.com/aspect-build/rules_lint/blob/v1.0.9/example/lint.sh
set -o errexit -o pipefail -o nounset

if [ "$#" -eq 0 ]; then
	echo "usage: lint.sh [target pattern...]"
	exit 1
fi

bazelrc=""

args=("--aspects=//tools:linters.bzl%ruff")

# NB: perhaps --remote_download_toplevel is needed as well with remote execution?
args+=(
	# Allow lints of code that fails some validation action
	# See https://github.com/aspect-build/rules_ts/pull/574#issuecomment-2073632879
	"--norun_validations"
	"--output_groups=rules_lint_human"
	"--remote_download_regex='.*AspectRulesLint.*'"
	"--@aspect_rules_lint//lint:fail_on_violation"
	"--keep_going"
)

while [[ "$#" -gt 0 ]]; do
	case "$1" in
		--bazelrc)
			if [[ "$#" -lt 2 ]]; then
				echo "Error: --bazelrc requires a path argument"
				exit 1
			fi
			bazelrc="--bazelrc=$2"
			shift 2
			;;
		--dry-run)
			fix="print"
			shift
			;;
		-*)
			echo "Unknown option: $1"
			exit 1
			;;
		*)
			# Not a flag, assume target patterns start here
			break
			;;
	esac
done

# Run linters
bazel ${bazelrc} build ${args[@]} $@
