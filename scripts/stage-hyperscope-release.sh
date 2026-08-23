#!/usr/bin/env bash
# Stage a clean Hyperscope bundle from an already validated Trunk build.
# Local heavyweight test models and retired matcap images are never copied.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="$repo_root/dist"
output_arg="${1:-dist-release}"

if [[ ! -f "$source_dir/index.html" ]]; then
    echo "error: $source_dir is not a completed Trunk build" >&2
    exit 2
fi

if [[ "$output_arg" = /* ]]; then
    output_dir="$(realpath -m "$output_arg")"
else
    output_dir="$(realpath -m "$repo_root/$output_arg")"
fi
source_dir="$(realpath "$source_dir")"

case "$output_dir" in
    "$repo_root"|"$source_dir"|"$source_dir"/*)
        echo "error: release output must be a new directory outside dist" >&2
        exit 2
        ;;
esac
if [[ -e "$output_dir" ]]; then
    echo "error: release output already exists: $output_dir" >&2
    exit 2
fi

output_parent="$(dirname "$output_dir")"
mkdir -p "$output_parent"
stage_dir="$(mktemp -d "$output_parent/.hyperscope-release.XXXXXX")"
cleanup() {
    rm -rf -- "$stage_dir"
}
trap cleanup EXIT

(
    cd "$source_dir"
    tar --exclude='./local-glbs' --exclude='./matcaps' -cf - .
) | (
    cd "$stage_dir"
    tar -xf -
)

if [[ -e "$stage_dir/local-glbs" || -e "$stage_dir/matcaps" ]]; then
    echo "error: excluded development assets entered the staged bundle" >&2
    exit 1
fi
if [[ ! -f "$stage_dir/ASSET_ATTRIBUTION.md" ]]; then
    echo "error: staged bundle is missing ASSET_ATTRIBUTION.md" >&2
    exit 1
fi

mv "$stage_dir" "$output_dir"
trap - EXIT
echo "Staged release bundle: $output_dir"
