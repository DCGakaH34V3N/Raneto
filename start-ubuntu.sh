#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="$SCRIPT_DIR/.raneto"
PID_FILE="$RUNTIME_DIR/raneto.pid"
LOG_FILE="$RUNTIME_DIR/raneto.log"
SECRET_FILE="$RUNTIME_DIR/session-secret"
INSTALL_MARKER="$RUNTIME_DIR/npm-install.complete"

ACTION="start"
BIND_ADDRESS="${ADDRESS:-0.0.0.0}"
PORT_NUMBER="${PORT:-9490}"
INSTALL_NODE=0

usage() {
  cat <<'EOF'
Run Raneto directly with Node.js (no Docker).

Usage:
  ./start-ubuntu.sh [start|stop|restart|status|logs] [options]

Options:
  --port PORT       Listening port (default: 9490)
  --bind ADDRESS    Listening address (default: 0.0.0.0)
  --local           Listen on 127.0.0.1 only
  --install-node    Install Node.js 24 from NodeSource when required
  -h, --help        Show this help

Environment variables such as SESSION_SECRET, ADMIN_USERNAME,
ADMIN_PASSWORD, ALLOW_EDITING, SITE_TITLE, and CONTENT_DIR are passed
through to Raneto. ALLOW_EDITING defaults to false in this script.
EOF
}

die() {
  echo "Error: $*" >&2
  exit 1
}

if (($#)) && [[ "$1" != -* ]]; then
  ACTION="$1"
  shift
fi

while (($#)); do
  case "$1" in
    --port)
      (($# >= 2)) || die "--port requires a value"
      PORT_NUMBER="$2"
      shift 2
      ;;
    --bind)
      (($# >= 2)) || die "--bind requires a value"
      BIND_ADDRESS="$2"
      shift 2
      ;;
    --local)
      BIND_ADDRESS="127.0.0.1"
      shift
      ;;
    --install-node)
      INSTALL_NODE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *) die "unknown option: $1" ;;
  esac
done

case "$ACTION" in
  start|stop|restart|status|logs) ;;
  *) die "unknown action: $ACTION" ;;
esac

[[ "$PORT_NUMBER" =~ ^[0-9]+$ ]] || die "port must be numeric"
((PORT_NUMBER >= 1 && PORT_NUMBER <= 65535)) || die "port must be between 1 and 65535"

mkdir -p "$RUNTIME_DIR"
chmod 700 "$RUNTIME_DIR"

read_pid() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(<"$PID_FILE")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  printf '%s' "$pid"
}

running_pid() {
  local pid
  pid="$(read_pid)" || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  printf '%s' "$pid"
}

runtime_is_supported() {
  command -v node >/dev/null 2>&1 || return 1
  command -v npm >/dev/null 2>&1 || return 1
  local node_version node_major node_minor npm_version npm_major npm_minor
  node_version="$(node --version)"
  node_version="${node_version#v}"
  IFS=. read -r node_major node_minor _ <<<"$node_version"
  ((node_major > 23 || (node_major == 23 && node_minor >= 14))) || return 1

  npm_version="$(npm --version)"
  IFS=. read -r npm_major npm_minor _ <<<"$npm_version"
  ((npm_major > 11 || (npm_major == 11 && npm_minor >= 9))) || return 1
}

install_node() {
  [[ "$(uname -s)" == "Linux" ]] || die "automatic Node.js installation is supported on Ubuntu only"
  [[ -r /etc/os-release ]] || die "cannot identify the operating system"
  # shellcheck disable=SC1091
  source /etc/os-release
  [[ "${ID:-}" == "ubuntu" ]] || die "automatic Node.js installation supports Ubuntu only (detected: ${ID:-unknown})"

  local -a sudo_cmd=()
  if ((EUID != 0)); then
    command -v sudo >/dev/null 2>&1 || die "sudo is required to install Node.js 24"
    sudo_cmd=(sudo)
  fi

  echo "Installing Node.js 24 from NodeSource..."
  "${sudo_cmd[@]}" apt-get update
  "${sudo_cmd[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl gnupg
  curl --fail --silent --show-error --location \
    https://deb.nodesource.com/setup_24.x -o "$RUNTIME_DIR/nodesource-setup.sh"
  "${sudo_cmd[@]}" bash "$RUNTIME_DIR/nodesource-setup.sh"
  "${sudo_cmd[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y nodejs
  hash -r
}

check_runtime() {
  if ! runtime_is_supported; then
    if ((INSTALL_NODE == 1)); then
      install_node
    else
      local node_version="not installed" npm_version="not installed"
      command -v node >/dev/null 2>&1 && node_version="$(node --version)"
      command -v npm >/dev/null 2>&1 && npm_version="$(npm --version)"
      die "unsupported runtime (Node.js $node_version, npm $npm_version); rerun with --install-node or install Node.js 24"
    fi
  fi

  runtime_is_supported || die "Node.js installation completed, but Node.js >=23.14.0 and npm >=11.9.0 are not active"
}

install_dependencies() {
  if [[ ! -d "$SCRIPT_DIR/node_modules" || ! -f "$INSTALL_MARKER" || \
        "$SCRIPT_DIR/package-lock.json" -nt "$INSTALL_MARKER" || \
        "$SCRIPT_DIR/package.json" -nt "$INSTALL_MARKER" ]]; then
    echo "Installing production dependencies..."
    (cd "$SCRIPT_DIR" && npm ci --omit=dev)
    touch "$INSTALL_MARKER"
  fi
}

ensure_secret() {
  if [[ -n "${SESSION_SECRET:-}" ]]; then
    return
  fi

  if [[ ! -s "$SECRET_FILE" ]]; then
    if command -v openssl >/dev/null 2>&1; then
      openssl rand -base64 48 >"$SECRET_FILE"
    else
      command -v base64 >/dev/null 2>&1 || die "openssl or base64 is required to generate SESSION_SECRET"
      head -c 48 /dev/urandom | base64 >"$SECRET_FILE"
    fi
    chmod 600 "$SECRET_FILE"
  fi
  SESSION_SECRET="$(<"$SECRET_FILE")"
  export SESSION_SECRET
}

start_service() {
  local pid check_host url
  if pid="$(running_pid)"; then
    echo "Raneto is already running (PID $pid)"
    return
  fi
  rm -f "$PID_FILE"

  check_runtime
  install_dependencies
  ensure_secret

  export NODE_ENV="${NODE_ENV:-production}"
  export ADDRESS="$BIND_ADDRESS"
  export PORT="$PORT_NUMBER"
  export ALLOW_EDITING="${ALLOW_EDITING:-false}"

  echo "Starting Raneto on $ADDRESS:$PORT..."
  (
    cd "$SCRIPT_DIR"
    nohup node server.js >>"$LOG_FILE" 2>&1 &
    echo "$!" >"$PID_FILE"
  )
  pid="$(<"$PID_FILE")"

  check_host="$BIND_ADDRESS"
  [[ "$check_host" == "0.0.0.0" ]] && check_host="127.0.0.1"
  url="http://$check_host:$PORT_NUMBER"
  if ! command -v curl >/dev/null 2>&1; then
    sleep 2
    if kill -0 "$pid" 2>/dev/null; then
      echo "Raneto is running at $url (PID $pid; HTTP check skipped because curl is unavailable)"
      echo "Log: $LOG_FILE"
      return
    fi
  fi
  for _ in {1..30}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "Raneto exited during startup. Recent log output:" >&2
      tail -n 50 "$LOG_FILE" >&2 || true
      rm -f "$PID_FILE"
      return 1
    fi
    if curl --fail --silent "$url" >/dev/null 2>&1; then
      echo "Raneto is running at $url (PID $pid)"
      echo "Log: $LOG_FILE"
      return
    fi
    sleep 1
  done

  echo "Raneto started (PID $pid), but the HTTP health check did not succeed within 30 seconds."
  echo "Inspect the log with: $0 logs"
}

stop_service() {
  local pid
  if ! pid="$(running_pid)"; then
    rm -f "$PID_FILE"
    echo "Raneto is not running"
    return
  fi

  echo "Stopping Raneto (PID $pid)..."
  kill "$pid"
  for _ in {1..20}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$PID_FILE"
      echo "Raneto stopped"
      return
    fi
    sleep 1
  done
  die "Raneto did not stop within 20 seconds (PID $pid)"
}

case "$ACTION" in
  start) start_service ;;
  stop) stop_service ;;
  restart)
    stop_service
    start_service
    ;;
  status)
    if pid="$(running_pid)"; then
      echo "Raneto is running (PID $pid)"
    else
      rm -f "$PID_FILE"
      echo "Raneto is not running"
      exit 1
    fi
    ;;
  logs)
    [[ -f "$LOG_FILE" ]] || die "log file does not exist yet: $LOG_FILE"
    tail -n 100 -f "$LOG_FILE"
    ;;
esac
