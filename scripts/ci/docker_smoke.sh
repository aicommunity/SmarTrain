#!/usr/bin/env bash
# CPU smoke: build image and run smartrain --help.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
IMAGE_TAG="${SMARTTRAIN_DOCKER_IMAGE:-smartrain:smoke}"
docker build -t "$IMAGE_TAG" .
docker run --rm "$IMAGE_TAG" --help
echo "[OK] docker smoke passed for $IMAGE_TAG"
