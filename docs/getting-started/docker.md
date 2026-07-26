> Russian version: [../ru/getting-started/docker.md](../ru/getting-started/docker.md)

# Docker

SmarTrain ships a root `Dockerfile` targeting **CUDA 12.8** runtime and the project **cu128** torch policy (pins via `ARG`/`ENV` in the Dockerfile).

## Build

```bash
docker build -t smartrain:local .
```

## Smoke (CPU)

```bash
docker run --rm smartrain:local --help
# or
bash scripts/ci/docker_smoke.sh
```

## GPU

```bash
docker run --rm --gpus all -v "$PWD/workspace:/workspace" \
  -e SMART_TRAIN_WORKSPACE=/workspace \
  smartrain:local train --help
```

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Notes

- `ENTRYPOINT` is `smartrain`; pass CLI args after the image name.
- Torch is installed from `https://download.pytorch.org/whl/cu128` with pinned versions in the Dockerfile (override with `--build-arg TORCH_VERSION=...`).
- `SMART_TRAIN_SKIP_TORCH_POLICY=1` inside the image avoids re-running `deps sync-torch` on every start.
