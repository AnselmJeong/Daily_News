#!/usr/bin/env bash
# Watch Articles/ for new/changed PDFs and trigger weekly_news a couple
# minutes AFTER activity settles (so we don't re-run mid-drag-and-drop).
#
# Requires fswatch:  brew install fswatch
#
# Usage:
#   ./watch_articles.sh &             # run in background
#   ./watch_articles.sh --foreground  # keep attached for debugging
#
# Cooldown (seconds) — how long to wait after last change before running.
# 120 = 2 minutes, which is enough for Zotero/Paperpile etc. to finish
# dropping files + letting index.json get updated.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEWS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ARTICLES_DIR="$(cd "${NEWS_DIR}/.." && pwd)"
COOLDOWN="${COOLDOWN:-120}"

if ! command -v fswatch >/dev/null 2>&1; then
  echo "fswatch not found. Install with:  brew install fswatch"
  exit 1
fi

echo "watching: ${ARTICLES_DIR}"
echo "cooldown: ${COOLDOWN}s"

# Debounced trigger loop
#  - fswatch emits on every change
#  - we bump a timestamp; a background "debouncer" checks every 10 s whether
#    the last event is older than COOLDOWN and if so runs the pipeline once.
LAST_EVENT_FILE="${NEWS_DIR}/.cache/last_event"
RAN_FOR_FILE="${NEWS_DIR}/.cache/last_run_for"
mkdir -p "${NEWS_DIR}/.cache"
date +%s > "${LAST_EVENT_FILE}"

debouncer() {
  while true; do
    sleep 10
    now=$(date +%s)
    last=$(cat "${LAST_EVENT_FILE}" 2>/dev/null || echo "${now}")
    ran=$(cat "${RAN_FOR_FILE}" 2>/dev/null || echo 0)
    delta=$((now - last))
    if [ "${delta}" -ge "${COOLDOWN}" ] && [ "${last}" -gt "${ran}" ]; then
      echo "[$(date '+%H:%M:%S')] quiet for ${delta}s since last change — running weekly_news"
      "${SCRIPT_DIR}/run_daily_news.sh"
      echo "${last}" > "${RAN_FOR_FILE}"
    fi
  done
}

debouncer &
DEBOUNCER_PID=$!
trap 'kill ${DEBOUNCER_PID} 2>/dev/null' EXIT

# Emit only events on .pdf files, ignore the _news output dir itself
fswatch \
  --exclude='/\._' \
  --exclude='/\.DS_Store$' \
  --exclude="${NEWS_DIR}" \
  --include='\.pdf$' \
  --event-flag-type \
  --format '%p' \
  "${ARTICLES_DIR}" \
| while IFS= read -r path; do
    echo "[$(date '+%H:%M:%S')] change: ${path}"
    date +%s > "${LAST_EVENT_FILE}"
  done
