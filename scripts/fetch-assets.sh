#!/usr/bin/env bash
#
# Download the optional glTF sample models used for material and transmission
# testing. These are not tracked in the repository because of their size; the
# demo pages run fine without them (they load the tracked `horse.glb` and
# `ant.glb`).
#
# Usage:
#   scripts/fetch-assets.sh            # download any that are missing
#   scripts/fetch-assets.sh --force    # re-download everything
#
# Assets come from KhronosGroup/glTF-Sample-Assets and carry their own
# licenses; see that repository for per-model attribution.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_URL="https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models"

MODELS=(
    DragonAttenuation
    IORTestGrid
    TransmissionTest
)

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
    FORCE=1
fi

if ! command -v curl >/dev/null 2>&1; then
    echo "error: curl is required" >&2
    exit 1
fi

cd "$REPO_ROOT"

for model in "${MODELS[@]}"; do
    dest="$model.glb"
    if [[ -f "$dest" && $FORCE -eq 0 ]]; then
        echo "skip  $dest (already present; use --force to re-download)"
        continue
    fi

    url="$BASE_URL/$model/glTF-Binary/$model.glb"
    echo "fetch $dest"
    if ! curl --fail --location --progress-bar --output "$dest.part" "$url"; then
        echo "error: failed to download $url" >&2
        rm -f "$dest.part"
        exit 1
    fi
    mv "$dest.part" "$dest"
done

echo
echo "Done. Note that food_cans.glb and sardine_tin.glb are not published"
echo "assets and cannot be fetched by this script."
