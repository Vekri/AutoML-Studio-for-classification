#!/usr/bin/env bash
# Mirror this repo to https://github.com/Vekri/Binary-Classification
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "Set GITHUB_TOKEN with push access to Vekri/Binary-Classification"
  exit 1
fi
git push "https://x-access-token:${GITHUB_TOKEN}@github.com/Vekri/Binary-Classification.git" HEAD:main --force
echo "Pushed to https://github.com/Vekri/Binary-Classification"
