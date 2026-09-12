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
WORKER_SCRIPT="worker.py"
PID_FILE=".deus.pid"
WORKER_PID_FILE=".deus-worker.pid"
FRONTEND_DIR="frontend"
LOG_DIR="storage/logs"
WORKER_LOG="$LOG_DIR/worker.log"
API_LOG="$LOG_DIR/api.log"
LOG_MAX_BYTES=$((20 * 1024 * 1024))
# Written once the off-width embedding reset has run; see
# reset_invalid_embeddings_once.
EMBEDDING_RESET_MARKER="storage/.embedding_width_reset_done"

# nohup, not setsid. nohup *execs* the program rather than forking, so `$!`
# remains the python PID the PID file has to hold. setsid forks whenever the
# caller is already a process-group leader — which a backgrounded command is — and
# the PID file would then name a wrapper that had already exited, leaving
# kill_running with nothing to stop.
if command -v nohup >/dev/null 2>&1; then
    DETACH="nohup"
else
    DETACH=""
fi

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

# ---- Network ----

# This watcher is the only part of Deus that talks to GitHub — nothing in
# main.py or worker.py does. Losing DNS on a phone is routine (WiFi drops,
# Doze, a tailnet flap), so an unreachable origin must never be fatal and must
# never reprint git's "could not resolve host" fatal once per poll. OFFLINE
# remembers the last state so only the transitions get logged.
OFFLINE=0

# git applies no network timeout of its own. On a half-open link — associated
# to WiFi with no route, or a captive portal — fetch blocks indefinitely and
# the watch loop stops ticking entirely. Abort a transfer stalled under
# 1 KB/s for 20s instead.
git_net() {
    git -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=20 "$@"
}

# Fetch, swallowing git's own stderr so an offline phone reports once rather
# than every cycle. Returns non-zero when origin is unreachable; every caller
# treats that as "skip this cycle", never as "stop".
try_fetch() {
    local errout
    if errout=$(git_net fetch origin "$BRANCH" --quiet 2>&1); then
        if [ "$OFFLINE" -eq 1 ]; then
            ok "Network is back — resuming update checks."
            OFFLINE=0
        fi
        return 0
    fi

    if [ "$OFFLINE" -eq 0 ]; then
        warn "Cannot reach origin/$BRANCH — ${errout%%$'\n'*}"
        warn "The app keeps running on the code already on disk; update checks"
        warn "resume by themselves once the network is back."
        OFFLINE=1
    fi
    return 1
}

# ---- Helpers ----

activate_venv() {
    if [ -d "$VENV_DIR" ]; then
        source "$VENV_DIR/bin/activate"
    else
        warn "No venv found. Running with system Python."
    fi
}

# Android suspends Termux under Doze once the screen goes off. The listening
# socket survives in the kernel, so the TCP handshake still completes and the
# browser sits waiting on a frozen process — which surfaces as a connection
# timeout rather than "refused". A wake lock is what keeps the app answering.
acquire_wake_lock() {
    if command -v termux-wake-lock >/dev/null 2>&1; then
        # Not `cmd && ok`: that makes the failure the function's exit status,
        # and this runs at top level under `set -e`, so a wake lock that
        # cannot be taken (CLI present, Termux:API app missing) would kill
        # the watcher before the app was ever started.
        if termux-wake-lock 2>/dev/null; then
            ok "Wake lock acquired (Doze suspension disabled)."
        else
            warn "termux-wake-lock failed — is the Termux:API *app* installed?"
            warn "Without it the app WILL be suspended by Doze."
        fi
    else
        warn "termux-wake-lock not found — the app WILL be suspended by Doze."
        warn "Fix: pkg install termux-api  (and install the Termux:API app)"
    fi
}

release_wake_lock() {
    if command -v termux-wake-unlock >/dev/null 2>&1; then
        termux-wake-unlock 2>/dev/null || true
    fi
}

# Read a single key out of .env, trimming a trailing CR (CRLF checkouts) and
# any surrounding quotes. The single-quote strip used to be written as
# ${path%'} , which bash parses as the start of a quoted string rather than a
# literal quote — so single-quoted values were never unwrapped at all.
env_value() {
    local key="$1" value=""
    [ -f ".env" ] || { echo ""; return; }
    value=$(sed -n "s/^[[:space:]]*${key}[[:space:]]*=[[:space:]]*//p" .env | tail -1)
    value=${value%$'\r'}
    value=${value%\"}; value=${value#\"}
    value=${value%\'}; value=${value#\'}
    echo "$value"
}

# Read DB_PATH out of .env so the cleanup below targets the database the app
# actually opens, rather than a hardcoded guess.
db_path_from_env() {
    local path
    path=$(env_value "DB_PATH")
    echo "${path:-storage/scrooge.db}"
}

# An instance listening on every interface with no passphrase starts up looking
# perfectly healthy — nothing in the normal logs says "wide open". Say it here.
check_access_control() {
    local host passphrase
    host=$(env_value "API_HOST")
    passphrase=$(env_value "DASHBOARD_PASSPHRASE")

    case "$host" in
        127.0.0.1|localhost|::1) return 0 ;;
    esac

    if [ -z "$passphrase" ]; then
        err "════════════════════════════════════════════════════════════"
        err " DASHBOARD_PASSPHRASE is empty and API_HOST is ${host:-0.0.0.0}."
        err " The dashboard will accept anyone who can reach port 8000,"
        err " including the endpoints that spend DeepSeek/Gemini credits"
        err " and the DELETE routes."
        err " Fix: set DASHBOARD_PASSPHRASE in .env, or API_HOST=127.0.0.1."
        err "════════════════════════════════════════════════════════════"
    fi
}

check_env_file() {
    if [ ! -f ".env" ]; then
        err "No .env file found! The app will likely fail to start."
        err "Copy .env.example to .env and fill in your API keys."
        if [ -f ".env.example" ]; then
            warn "Run: cp .env.example .env  (then edit with your keys)"
        fi
        return
    fi
    check_access_control
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

    # Split the two failures: "cannot resolve registry.npmjs.org" is an
    # offline phone, a build error is broken code. They need different fixes,
    # and neither may stop the app from starting.
    if ! (cd "$FRONTEND_DIR" && npm install); then
        err "npm install failed (offline?) — keeping the existing static export."
        return 1
    fi

    if ! (cd "$FRONTEND_DIR" && npm run build:static); then
        err "Frontend build failed! The backend will serve whatever is in out/."
        return 1
    fi

    ok "Frontend build complete."
}

stop_pid_file() {
    local pid_file="$1" label="$2"
    [ -f "$pid_file" ] || return 0

    local old_pid
    old_pid=$(cat "$pid_file")
    if kill -0 "$old_pid" 2>/dev/null; then
        log "Stopping $label (PID $old_pid)..."
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
            warn "Force killing $label (PID $old_pid)..."
            kill -9 "$old_pid" 2>/dev/null || true
        fi
    fi
    rm -f "$pid_file"
}

kill_running() {
    # API first: it only reads, so stopping it before the worker avoids
    # serving a half-written cycle.
    stop_pid_file "$PID_FILE" "$MAIN_SCRIPT"
    stop_pid_file "$WORKER_PID_FILE" "$WORKER_SCRIPT"
}

# Keep a log file from growing without bound. Three generations of 20 MB is
# enough to cover a few days of a phone's worth of output while staying far
# inside the storage a Termux install can spare; an unrotated worker log reached
# 246 MB on its own.
rotate_log() {
    local file="$1" size=0
    [ -f "$file" ] || return 0
    # stat's flags differ between GNU coreutils and Android/BSD; wc is the
    # portable fallback and is only ever run once per restart.
    size=$(stat -c %s "$file" 2>/dev/null || stat -f %z "$file" 2>/dev/null || wc -c < "$file")
    [ "${size:-0}" -gt "$LOG_MAX_BYTES" ] || return 0
    rm -f "${file}.3"
    [ -f "${file}.2" ] && mv "${file}.2" "${file}.3"
    [ -f "${file}.1" ] && mv "${file}.1" "${file}.2"
    mv "$file" "${file}.1"
    log "Rotated $(basename "$file") (was $size bytes)."
}

# One-shot repair, not a startup chore.
#
# This reset ran on EVERY restart. With the embedder discarding off-width
# vectors, that made it a self-replenishing embedding backlog: anything the
# producer got wrong was nulled, re-embedded, nulled again. The producer side is
# fixed (the embedder rejects rather than stores an off-width vector), so this
# only needs to clean up what was stored before — once. The marker file is what
# makes it once.
reset_invalid_embeddings_once() {
    if [ -f "$EMBEDDING_RESET_MARKER" ]; then
        return 0
    fi
    log "Clearing embeddings with a wrong vector width (one-time)..."
    # Path comes from .env. This used to be hardcoded to storage/deus.db while
    # .env pointed at storage/scrooge.db, so the cleanup silently did nothing
    # and created an empty database alongside the real one.
    DEUS_DB_PATH="$(db_path_from_env)" python - <<'PY' || true
import os
import sqlite3

db_path = os.environ.get("DEUS_DB_PATH", "storage/scrooge.db")
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='articles'"
    )
    if cur.fetchone():
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM articles WHERE length(embedding) != 12288"
            ).fetchone()[0]
            conn.execute(
                "UPDATE articles SET embedding = NULL WHERE length(embedding) != 12288"
            )
            conn.commit()
            print(f"Removed {count} invalid embeddings.")
        except Exception as e:
            print(f"Embedding cleanup skipped: {e}")
    conn.close()
PY
    mkdir -p "$(dirname "$EMBEDDING_RESET_MARKER")"
    date -u '+%Y-%m-%dT%H:%M:%SZ' > "$EMBEDDING_RESET_MARKER"
}

start_app() {
    check_env_file
    activate_venv
    reset_invalid_embeddings_once

    # Both processes used to be backgrounded with no redirection at all, so
    # structlog's output existed only in Termux scrollback: the moment the SSH
    # session or the Termux session that started deploy.sh went away, every log
    # line the app had produced was gone. That is why the classifier stall had to
    # be diagnosed out of the database rather than the logs.
    mkdir -p "$LOG_DIR"
    rotate_log "$WORKER_LOG"
    rotate_log "$API_LOG"

    # Two processes. The pipeline and Telegram bot run in the worker so they
    # can never stall the API's event loop; the API is read-mostly and stays
    # responsive while a cycle is running. They share state through SQLite
    # (WAL) and the sse_events outbox table.
    #
    # $DETACH (nohup) detaches them from the controlling terminal, so closing
    # the SSH session that started the watcher does not SIGHUP the app out from
    # under it.
    log "Starting $WORKER_SCRIPT (logging to $WORKER_LOG)..."
    $DETACH python "$WORKER_SCRIPT" >> "$WORKER_LOG" 2>&1 &
    local worker_pid=$!
    echo "$worker_pid" > "$WORKER_PID_FILE"
    ok "Started $WORKER_SCRIPT (PID $worker_pid)"

    log "Starting $MAIN_SCRIPT (logging to $API_LOG)..."
    $DETACH python "$MAIN_SCRIPT" >> "$API_LOG" 2>&1 &
    local new_pid=$!
    echo "$new_pid" > "$PID_FILE"
    ok "Started $MAIN_SCRIPT (PID $new_pid)"
}

install_deps_if_changed() {
    # Check if requirements.txt changed by comparing with our backup
    if ! cmp -s requirements.txt .requirements.txt.bak 2>/dev/null; then
        warn "requirements.txt changed (or first run) — reinstalling dependencies..."
        activate_venv
        # Stamp the backup only on success. Copying it unconditionally marks a
        # failed offline install as done, so the next cycle skips it and the
        # app starts with packages that were never installed.
        if pip install -r requirements.txt --quiet; then
            cp requirements.txt .requirements.txt.bak 2>/dev/null || true
            ok "Dependencies updated."
        else
            err "pip install failed (offline?) — dependency stamp left alone so the next cycle retries."
            return 1
        fi
    fi
}

pull_and_restart() {
    log "Pulling latest changes from origin/$BRANCH..."

    # Backup the current requirements.txt to detect changes
    cp requirements.txt .requirements.txt.bak 2>/dev/null || true

    try_fetch || {
        warn "Skipping this deploy cycle — the running app is untouched."
        return 1
    }

    if ! git reset --hard "origin/$BRANCH" 2>/dev/null; then
        err "git reset --hard failed. There may be local conflicts."
        err "Try running: git stash && git reset --hard origin/$BRANCH"
        return 1
    fi

    # Neither may block the restart. Exiting here would leave the phone with
    # the old processes killed and nothing serving; starting on the packages
    # and static export already on disk is strictly better than nothing.
    install_deps_if_changed || warn "Starting with the packages already installed."
    build_frontend || warn "Starting with the previous static export."

    kill_running
    start_app
}

check_for_updates() {
    # Fetch without merging. Offline is an expected state on a phone, so this
    # returns quietly rather than letting git's fatal reach the log.
    try_fetch || return 1

    local local_hash remote_hash
    local_hash=$(git rev-parse HEAD 2>/dev/null || echo "")
    remote_hash=$(git rev-parse "origin/$BRANCH" 2>/dev/null || echo "")

    # An empty remote hash means the tracking ref is missing, not that a new
    # commit landed. Comparing it against HEAD would fire a deploy on every
    # single poll.
    if [ -z "$local_hash" ] || [ -z "$remote_hash" ]; then
        warn "Could not resolve HEAD or origin/$BRANCH — skipping this check."
        return 1
    fi

    if [ "$local_hash" != "$remote_hash" ]; then
        ok "New commit detected!"
        log "  Local:  ${local_hash:0:8}"
        log "  Remote: ${remote_hash:0:8}"
        # Report the deploy's real outcome — returning 0 unconditionally
        # printed "Deploy complete" for cycles that never deployed anything.
        pull_and_restart || return 1
        return 0
    fi
    return 1
}

cleanup() {
    log "Shutting down watcher..."
    kill_running
    release_wake_lock
    exit 0
}

# ---- Main ----

trap cleanup SIGINT SIGTERM

cd "$(dirname "$0")"
log "Deus — Auto-Deploy Watcher"
log "Tracking: origin/$BRANCH"

acquire_wake_lock

# --once mode: single pull + restart, then exit
if [ "${1:-}" = "--once" ]; then
    if pull_and_restart; then
        log "One-shot deploy complete. Exiting."
    else
        warn "No update applied — starting on the code already on disk."
        kill_running
        start_app
        log "One-shot start complete. Exiting."
    fi
    exit 0
fi

# A non-numeric interval makes `sleep` fail, and the sleep sits in the watch
# loop's body where `set -e` still applies — the watcher would exit after one
# tick with no explanation.
case "$POLL_INTERVAL" in
    ''|*[!0-9]*)
        warn "Invalid poll interval '$POLL_INTERVAL' — falling back to 30s."
        POLL_INTERVAL=30
        ;;
esac

log "Poll interval: ${POLL_INTERVAL}s"
echo ""

# Initial start. Nothing here may abort the script: this runs at top level,
# where `set -e` is live, so a failed pip or npm — the normal outcome when the
# phone has no DNS — would exit deploy.sh before start_app ever ran, and the
# app would simply never come up. Both steps are best-effort by design.
install_deps_if_changed || warn "Continuing with the packages already installed."
build_frontend || warn "Continuing with the previous static export."
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
