#!/bin/bash
# ============================================================
#  Deus — Auto-Deploy Watcher (Termux)
#
#  Polls GitHub for new commits every N seconds.
#  When a change is detected, pulls the latest code and
#  restarts main.py automatically.
#
#  Usage:
#    chmod +x deploy.sh
#    ./deploy.sh              # default: check every 30s
#    ./deploy.sh 60           # check every 60s
#    ./deploy.sh --once       # pull + restart once, then exit
# ============================================================

set -euo pipefail

POLL_INTERVAL="${1:-30}"       # seconds between checks (default 30)
BRANCH="main"                  # git branch to track
VENV_DIR="venv"
MAIN_SCRIPT="main.py"
PID_FILE=".deus.pid"
FRONTEND_DIR="frontend"

# ---- Colors ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log()  { echo -e "${CYAN}[deploy $(date '+%H:%M:%S')]${NC} $*"; }
ok()   { echo -e "${GREEN}[deploy $(date '+%H:%M:%S')]${NC} $*"; }
warn() { echo -e "${YELLOW}[deploy $(date '+%H:%M:%S')]${NC} $*"; }
err()  { echo -e "${RED}[deploy $(date '+%H:%M:%S')]${NC} $*"; }

# ---- Helpers ----

activate_venv() {
    if [ -d "$VENV_DIR" ]; then
        source "$VENV_DIR/bin/activate"
    else
        warn "No venv found. Running with system Python."
    fi
}

check_env_file() {
    if [ ! -f ".env" ]; then
        err "No .env file found! The app will likely fail to start."
        err "Copy .env.example to .env and fill in your API keys."
        if [ -f ".env.example" ]; then
            warn "Run: cp .env.example .env  (then edit with your keys)"
        fi
    fi
}

build_frontend() {
    log "Checking frontend..."

    if [ ! -d "$FRONTEND_DIR" ]; then
        warn "No frontend/ directory found — skipping frontend build."
        return 0
    fi

    if [ ! -f "$FRONTEND_DIR/package.json" ]; then
        warn "No frontend/package.json — skipping frontend build."
        return 0
    fi

    if ! command -v node &>/dev/null; then
        warn "Node.js not found — skipping frontend build."
        warn "Install Node.js: pkg install nodejs"
        return 0
    fi

    if ! command -v npm &>/dev/null; then
        warn "npm not found — skipping frontend build."
        return 0
    fi

    # Always rebuild to ensure freshness
    warn "Building frontend static export..."

    (cd "$FRONTEND_DIR" && npm install && npm run build:static) || {
        err "Frontend build failed! The backend will serve whatever is in out/."
        return 1
    }

    ok "Frontend build complete."
}

kill_running() {
    if [ -f "$PID_FILE" ]; then
        local old_pid
        old_pid=$(cat "$PID_FILE")
        if kill -0 "$old_pid" 2>/dev/null; then
            log "Stopping running instance (PID $old_pid)..."
            kill "$old_pid" 2>/dev/null || true
            # Wait up to 5 seconds for graceful shutdown
            for i in $(seq 1 10); do
                if ! kill -0 "$old_pid" 2>/dev/null; then
                    break
                fi
                sleep 0.5
            done
            # Force kill if still alive
            if kill -0 "$old_pid" 2>/dev/null; then
                warn "Force killing PID $old_pid..."
                kill -9 "$old_pid" 2>/dev/null || true
            fi
        fi
        rm -f "$PID_FILE"
    fi
}

start_app() {
    check_env_file
    activate_venv
    log "Cleaning up invalid embeddings from database..."
    python -c "
import sqlite3, os
db_path = 'storage/deus.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name='articles'\")
    if cur.fetchone():
        try:
            count = conn.execute('SELECT COUNT(*) FROM articles WHERE length(embedding) != 12288').fetchone()[0]
            conn.execute('UPDATE articles SET embedding = NULL WHERE length(embedding) != 12288')
            conn.commit()
            if count > 0:
                print(f'Removed {count} invalid embeddings.')
        except Exception as e:
            print(f'Embedding cleanup skipped: {e}')
    conn.close()
" || true
    log "Starting $MAIN_SCRIPT..."
    python "$MAIN_SCRIPT" &
    local new_pid=$!
    echo "$new_pid" > "$PID_FILE"
    ok "Started $MAIN_SCRIPT (PID $new_pid)"
}

install_deps_if_changed() {
    # Check if requirements.txt changed by comparing with our backup
    if ! cmp -s requirements.txt .requirements.txt.bak 2>/dev/null; then
        warn "requirements.txt changed (or first run) — reinstalling dependencies..."
        activate_venv
        pip install -r requirements.txt --quiet
        cp requirements.txt .requirements.txt.bak 2>/dev/null || true
        ok "Dependencies updated."
    fi
}

pull_and_restart() {
    log "Pulling latest changes from origin/$BRANCH..."

    # Backup the current requirements.txt to detect changes
    cp requirements.txt .requirements.txt.bak 2>/dev/null || true

    git fetch origin "$BRANCH" --quiet || {
        err "git fetch failed — skipping this deploy cycle."
        return 1
    }

    if ! git reset --hard "origin/$BRANCH" 2>/dev/null; then
        err "git reset --hard failed. There may be local conflicts."
        err "Try running: git stash && git reset --hard origin/$BRANCH"
        return 1
    fi

    install_deps_if_changed
    build_frontend

    kill_running
    start_app
}

check_for_updates() {
    # Fetch without merging
    git fetch origin "$BRANCH" --quiet

    local local_hash remote_hash
    local_hash=$(git rev-parse HEAD)
    remote_hash=$(git rev-parse "origin/$BRANCH")

    if [ "$local_hash" != "$remote_hash" ]; then
        ok "New commit detected!"
        log "  Local:  ${local_hash:0:8}"
        log "  Remote: ${remote_hash:0:8}"
        pull_and_restart
        return 0
    fi
    return 1
}

cleanup() {
    log "Shutting down watcher..."
    kill_running
    exit 0
}

# ---- Main ----

trap cleanup SIGINT SIGTERM

cd "$(dirname "$0")"
log "Deus — Auto-Deploy Watcher"
log "Tracking: origin/$BRANCH"

# --once mode: single pull + restart, then exit
if [ "${1:-}" = "--once" ]; then
    pull_and_restart
    log "One-shot deploy complete. Exiting."
    exit 0
fi

log "Poll interval: ${POLL_INTERVAL}s"
echo ""

# Initial start
install_deps_if_changed
build_frontend
kill_running
start_app
echo ""

# Watch loop
while true; do
    if check_for_updates; then
        ok "Deploy complete. Watching for next change..."
    fi
    sleep "$POLL_INTERVAL"
done
