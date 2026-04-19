#!/usr/bin/env bash
# Wrapper invoked by launchd / fswatch / cron.
# Keeps stdout/stderr in _news/.cache/run.log for debugging.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEWS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ARTICLES_ROOT="$(cd "${NEWS_DIR}/.." && pwd)"
LOG="${NEWS_DIR}/.cache/run.log"
mkdir -p "${NEWS_DIR}/.cache"

# Prefer the installed `daily-news` entry point (uv tool install).
# Fall back to `uv tool run` or `python -m daily_news` if not on PATH.
if command -v daily-news >/dev/null 2>&1; then
  CMD=(daily-news)
elif command -v uv >/dev/null 2>&1; then
  CMD=(uv tool run --from daily-news daily-news)
else
  CMD=("${PYTHON:-python3}" -m daily_news)
fi

{
  echo ""
  echo "=== $(date '+%Y-%m-%d %H:%M:%S')  daily_news launching ==="
  echo "cmd: ${CMD[*]}"
  "${CMD[@]}" --articles-root "${ARTICLES_ROOT}" "$@"
  rc=$?
  echo "exit: ${rc}"
} >> "${LOG}" 2>&1
exit ${rc}
