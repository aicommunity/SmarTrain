> English version: [../../getting-started/docker.md](../../getting-started/docker.md)

# Docker

В корне репозитория есть `Dockerfile` под runtime **CUDA 12.8** и политику torch **cu128** (пины через `ARG`/`ENV` в Dockerfile).

## Сборка

```bash
docker build -t smartrain:local .
```

## Smoke (CPU)

```bash
docker run --rm smartrain:local --help
# или
bash scripts/ci/docker_smoke.sh
```

## GPU

```bash
docker run --rm --gpus all -v "$PWD/workspace:/workspace" \
  -e SMART_TRAIN_WORKSPACE=/workspace \
  smartrain:local train --help
```

Нужен [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Заметки

- `ENTRYPOINT` — `smartrain`; аргументы CLI после имени образа.
- Torch ставится с `https://download.pytorch.org/whl/cu128` с пинами в Dockerfile (`--build-arg TORCH_VERSION=...`).
- В образе `SMART_TRAIN_SKIP_TORCH_POLICY=1`, чтобы не гонять `deps sync-torch` на каждый старт.
