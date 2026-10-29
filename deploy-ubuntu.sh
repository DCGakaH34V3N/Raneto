#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="raneto"
IMAGE_NAME="raneto-local:latest"
BIND_ADDRESS="0.0.0.0"
INSTALL_DOCKER=1
HOST_PORT="9490"

usage() {
  cat <<'EOF'
Deploy the current Raneto checkout on Ubuntu with Docker.

Usage:
  sudo ./deploy-ubuntu.sh [options]

Options:
  --port PORT             Host port (default: 9490)
  --bind ADDRESS          Bind address (default: 0.0.0.0)
  --local                 Bind to 127.0.0.1 only
  --name NAME             Container name (default: raneto)
  --image NAME            Docker image name (default: raneto-local:latest)
  --skip-docker-install   Fail instead of installing Docker when unavailable
  -h, --help              Show this help

Examples:
  sudo ./deploy-ubuntu.sh
  sudo ./deploy-ubuntu.sh --port 9000
  sudo ./deploy-ubuntu.sh --local --port 9490
EOF
}

die() { echo "Error: $*" >&2; exit 1; }

while (($#)); do
  case "$1" in
    --port) (($# >= 2)) || die "--port requires a value"; HOST_PORT="$2"; shift 2 ;;
    --bind) (($# >= 2)) || die "--bind requires a value"; BIND_ADDRESS="$2"; shift 2 ;;
    --local) BIND_ADDRESS="127.0.0.1"; shift ;;
    --name) (($# >= 2)) || die "--name requires a value"; CONTAINER_NAME="$2"; shift 2 ;;
    --image) (($# >= 2)) || die "--image requires a value"; IMAGE_NAME="$2"; shift 2 ;;
    --skip-docker-install) INSTALL_DOCKER=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

[[ "$HOST_PORT" =~ ^[0-9]+$ ]] || die "port must be numeric"
((HOST_PORT >= 1 && HOST_PORT <= 65535)) || die "port must be between 1 and 65535"
[[ -f "$SCRIPT_DIR/Dockerfile" ]] || die "Dockerfile not found in $SCRIPT_DIR"
[[ -d "$SCRIPT_DIR/content" ]] || die "content directory not found"

[[ -r /etc/os-release ]] || die "cannot identify the operating system"
# shellcheck disable=SC1091
source /etc/os-release
[[ "${ID:-}" == "ubuntu" ]] || die "this script supports Ubuntu only (detected: ${ID:-unknown})"

if ((EUID == 0)); then SUDO=();
elif command -v sudo >/dev/null 2>&1; then SUDO=(sudo);
else die "run as root or install sudo"; fi

if ! command -v docker >/dev/null 2>&1; then
  ((INSTALL_DOCKER == 1)) || die "Docker is not installed"
  echo "Installing Docker from Ubuntu repositories..."
  "${SUDO[@]}" apt-get update
  DEBIAN_FRONTEND=noninteractive "${SUDO[@]}" apt-get install -y docker.io ca-certificates curl
fi

if command -v systemctl >/dev/null 2>&1; then "${SUDO[@]}" systemctl enable --now docker; fi
if docker info >/dev/null 2>&1; then DOCKER=(docker);
elif "${SUDO[@]}" docker info >/dev/null 2>&1; then DOCKER=("${SUDO[@]}" docker);
else die "Docker daemon is unavailable"; fi

echo "Building $IMAGE_NAME..."
"${DOCKER[@]}" build --pull --tag "$IMAGE_NAME" "$SCRIPT_DIR"

if "${DOCKER[@]}" container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "Replacing existing container $CONTAINER_NAME..."
  "${DOCKER[@]}" rm --force "$CONTAINER_NAME" >/dev/null
fi

echo "Starting $CONTAINER_NAME..."
"${DOCKER[@]}" run --detach \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  --publish "$BIND_ADDRESS:$HOST_PORT:8080" \
  --env ADDRESS=0.0.0.0 \
  --env PORT=8080 \
  --env ALLOW_EDITING=false \
  --volume "$SCRIPT_DIR/content:/opt/raneto/content" \
  --volume "$SCRIPT_DIR/config/config.js:/opt/raneto/config/config.js:ro" \
  "$IMAGE_NAME" >/dev/null

if ! command -v curl >/dev/null 2>&1; then
  "${SUDO[@]}" apt-get update
  DEBIAN_FRONTEND=noninteractive "${SUDO[@]}" apt-get install -y curl
fi

check_host="$BIND_ADDRESS"
[[ "$check_host" == "0.0.0.0" ]] && check_host="127.0.0.1"
url="http://$check_host:$HOST_PORT"
for _ in {1..30}; do
  if curl --fail --silent "$url" >/dev/null 2>&1; then
    echo "Raneto is running at $url"
    echo "Container: ${DOCKER[*]} logs --follow $CONTAINER_NAME"
    [[ "$BIND_ADDRESS" != "0.0.0.0" ]] || echo "Public bind enabled on port $HOST_PORT. Configure TLS and authentication before enabling editing."
    exit 0
  fi
  sleep 1
done

echo "Raneto did not become healthy. Recent logs:" >&2
"${DOCKER[@]}" logs --tail 100 "$CONTAINER_NAME" >&2 || true
exit 1
