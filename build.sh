#!/bin/bash
#
# Builds alphafold3.sif from the official AlphaFold 3 repository, following the upstream instructions
# (https://github.com/google-deepmind/alphafold3/blob/main/docs/installation.md): build the Docker image from
# docker/Dockerfile, then convert it to a Singularity image.
#
# Usage: ./build.sh [git ref]
#   git ref: the AlphaFold 3 tag, branch, or commit to build. Defaults to the latest release tag.
#
# Requires docker or podman (set CONTAINER_TOOL to choose) and singularity (or apptainer).

set -euo pipefail

AF3_REPO=https://github.com/google-deepmind/alphafold3.git
BUILD_DIR=$(cd "$(dirname "$0")" && pwd)
AF3_SRC="$BUILD_DIR/alphafold3_src"
SIF="$BUILD_DIR/alphafold3.sif"

# Pick the container build tool
if [[ -z "${CONTAINER_TOOL:-}" ]]; then
  if command -v docker >/dev/null; then
    CONTAINER_TOOL=docker
  elif command -v podman >/dev/null; then
    CONTAINER_TOOL=podman
  else
    echo "Neither docker nor podman was found - one is required to build the AlphaFold 3 image." >&2
    exit 1
  fi
fi
if command -v singularity >/dev/null; then
  SINGULARITY=singularity
elif command -v apptainer >/dev/null; then
  SINGULARITY=apptainer
else
  echo "Neither singularity nor apptainer was found." >&2
  exit 1
fi

# Fetch or update the AlphaFold 3 source
if [[ -d "$AF3_SRC/.git" ]]; then
  echo "Updating AlphaFold 3 source in $AF3_SRC"
  git -C "$AF3_SRC" fetch --tags --force origin
else
  echo "Cloning AlphaFold 3 source into $AF3_SRC"
  git clone "$AF3_REPO" "$AF3_SRC"
fi

AF3_REF=${1:-$(git -C "$AF3_SRC" tag --list 'v*' --sort=-v:refname | head -n 1)}
# Prefer the remote branch, so that building a branch name picks up the latest fetched commits
if git -C "$AF3_SRC" rev-parse --verify --quiet "origin/$AF3_REF^{commit}" >/dev/null; then
  git -C "$AF3_SRC" checkout --quiet --detach "origin/$AF3_REF"
else
  git -C "$AF3_SRC" checkout --quiet --detach "$AF3_REF"
fi
AF3_COMMIT=$(git -C "$AF3_SRC" rev-parse --short HEAD)
IMAGE_TAG="alphafold3:${AF3_REF//\//-}"
echo "Building AlphaFold 3 $AF3_REF ($AF3_COMMIT) as $IMAGE_TAG using $CONTAINER_TOOL"

# Keep the large Singularity cache and temporary files out of /tmp
mkdir -p "$BUILD_DIR/singularity/cache" "$BUILD_DIR/singularity/tmp"
export SINGULARITY_CACHEDIR="$BUILD_DIR/singularity/cache"
export APPTAINER_CACHEDIR="$BUILD_DIR/singularity/cache"
export SINGULARITY_TMPDIR="$BUILD_DIR/singularity/tmp"
export APPTAINER_TMPDIR="$BUILD_DIR/singularity/tmp"

# Build the image from the upstream Dockerfile. The raised file descriptor limit is recommended upstream for
#  RHEL-like systems, and is harmless elsewhere.
"$CONTAINER_TOOL" build --ulimit nofile=65535:65535 -t "$IMAGE_TAG" -f "$AF3_SRC/docker/Dockerfile" "$AF3_SRC"

# Convert the image to a Singularity image via an archive (works with both docker and podman, and doesn't need a
#  local registry). Build to a temporary file so that a failed build doesn't clobber an existing container.
ARCHIVE="$BUILD_DIR/singularity/tmp/alphafold3.tar"
trap 'rm -f "$ARCHIVE" "$SIF.tmp"' EXIT
"$CONTAINER_TOOL" save -o "$ARCHIVE" "$IMAGE_TAG"
"$SINGULARITY" build --force "$SIF.tmp" "docker-archive://$ARCHIVE"

# Sanity check that AlphaFold 3 is importable in the new container
"$SINGULARITY" exec "$SIF.tmp" python -c 'import alphafold3'

mv "$SIF.tmp" "$SIF"
echo "Built $SIF from AlphaFold 3 $AF3_REF ($AF3_COMMIT)"
