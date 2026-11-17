#!/usr/bin/env bash
# Isolate from ambient credentials and use only disposable local state.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
tf_bin="$(command -v terraform)"
plan_home="$(mktemp -d)"
trap 'rm -rf "$plan_home"' EXIT
env -i PATH="$PATH" HOME="$plan_home" AWS_EC2_METADATA_DISABLED=true \
  "$tf_bin" -chdir="$root/infra" plan -refresh=false -input=false -lock=false -var-file=offline.tfvars "$@"
