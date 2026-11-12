#!/usr/bin/env bash
# Reproduce the complete frozen evaluation from a checkout with Docker.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

backend="${1:-cpu}"
config="${2:-configs/yolo26s_mvrsd_vis.yaml}"
case "$backend" in
  cpu) device=cpu ;;
  cu128) device=cuda ;;
  *) echo "Usage: $0 [cpu|cu128] [configs/model.yaml]" >&2; exit 2 ;;
esac
if [[ ! -f "$config" ]]; then
  echo "Configuration not found: $config" >&2
  exit 2
fi

image="aerial-vehicles:$backend"
docker build --build-arg "TORCH_BACKEND=$backend" -t "$image" .
mkdir -p data/forest_vehicles artifacts/cache artifacts/reference_indexes outputs

run=(--rm --user "$(id -u):$(id -g)"
  --mount "type=bind,src=$PWD/data,dst=/app/data"
  --mount "type=bind,src=$PWD/artifacts,dst=/app/artifacts"
  --mount "type=bind,src=$PWD/outputs,dst=/app/outputs")
gpu=()
if [[ "$device" == cuda ]]; then
  gpu+=(--gpus all)
fi
credentials=()
if [[ -f .env ]]; then
  credentials+=(--env-file .env)
fi

docker run "${run[@]}" "${gpu[@]}" "${credentials[@]}" "$image" download-dataset forest
docker run "${run[@]}" "${gpu[@]}" "$image" prepare --config "$config" --device "$device"
docker run "${run[@]}" "${gpu[@]}" "$image" index --config "$config" --device "$device"
# Evaluation only reads the dataset. Each run gets a new output directory.
docker run --rm --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$PWD/data,dst=/app/data,readonly" \
  --mount "type=bind,src=$PWD/artifacts,dst=/app/artifacts" \
  --mount "type=bind,src=$PWD/outputs,dst=/app/outputs" \
  "${gpu[@]}" "$image" evaluate --config "$config" --device "$device" --splits all
