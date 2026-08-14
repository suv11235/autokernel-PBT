#!/usr/bin/env bash
# Create a commit without Cursor-injected Co-authored-by trailers.
# Usage: scripts/git_commit_clean.sh -m "subject" [-m "body"...]
#        scripts/git_commit_clean.sh -F message.txt
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! git diff --cached --quiet; then
  :
elif [ -z "$(git diff --cached --name-only)" ]; then
  echo "error: nothing staged; run git add first" >&2
  exit 1
fi

MSG_FILE=$(mktemp)
trap 'rm -f "$MSG_FILE"' EXIT

if [ "${1:-}" = "-F" ] && [ -n "${2:-}" ]; then
  sed '/^Co-authored-by: Cursor/d' "$2" > "$MSG_FILE"
else
  git interpret-trailers --parse <<<"" >/dev/null 2>&1 || true
  : > "$MSG_FILE"
  while [ $# -gt 0 ]; do
    case "$1" in
      -m)
        shift
        printf '%s\n' "$1" >> "$MSG_FILE"
        ;;
      *)
        echo "usage: $0 -m msg [-m body...] | -F file" >&2
        exit 1
        ;;
    esac
    shift
  done
fi

if [ ! -s "$MSG_FILE" ]; then
  echo "error: empty commit message" >&2
  exit 1
fi

if grep -qi 'co-authored-by:.*cursor' "$MSG_FILE"; then
  echo "error: message still contains Cursor co-author" >&2
  exit 1
fi

TREE=$(git write-tree)
PARENT=$(git rev-parse --verify HEAD 2>/dev/null || true)
if [ -n "$PARENT" ]; then
  NEW=$(git commit-tree "$TREE" -p "$PARENT" -F "$MSG_FILE")
else
  NEW=$(git commit-tree "$TREE" -F "$MSG_FILE")
fi
git reset --soft "$NEW"

echo "Created commit $(git rev-parse --short HEAD)"
git log -1 --format=fuller

if git log -1 --format=%B | grep -qiE 'co-authored-by:.*cursor'; then
  echo "error: Cursor co-author trailer detected after commit" >&2
  exit 1
fi
