#!/usr/bin/env bash
# Create a commit without Cursor-injected Co-authored-by trailers.
# Usage: scripts/git_commit_clean.sh -m "subject" [-m "body"...]
#        scripts/git_commit_clean.sh -F message.txt
#
# -m and -F are mutually exclusive: git honours both, but this helper used to keep the
# file and drop every -m without a word, so it now refuses the combination instead.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# On a detached HEAD, `git reset --soft` below moves HEAD and no branch: the commit
# is reachable only from the reflog, and the next checkout orphans it. That happened
# once. Refuse up front -- before anything is written.
if ! git symbolic-ref -q HEAD >/dev/null; then
  echo "error: HEAD is detached; check out a branch first (git switch <branch>)" >&2
  exit 1
fi

if ! git diff --cached --quiet; then
  :
elif [ -z "$(git diff --cached --name-only)" ]; then
  echo "error: nothing staged; run git add first" >&2
  exit 1
fi

MSG_FILE=$(mktemp)
trap 'rm -f "$MSG_FILE"' EXIT

usage() {
  echo "usage: $0 -m msg [-m body...] | -F file" >&2
  exit 1
}

if [ "${1:-}" = "-F" ]; then
  if [ $# -ne 2 ]; then
    echo "error: -F takes exactly one file and cannot be combined with -m" >&2
    usage
  fi
  if [ ! -f "$2" ]; then
    echo "error: message file not found: $2" >&2
    exit 1
  fi
  sed '/^Co-authored-by: Cursor/d' "$2" > "$MSG_FILE"
else
  git interpret-trailers --parse <<<"" >/dev/null 2>&1 || true
  : > "$MSG_FILE"
  FIRST_M=1
  while [ $# -gt 0 ]; do
    case "$1" in
      -m)
        shift
        if [ $# -eq 0 ]; then
          echo "error: -m needs a message" >&2
          usage
        fi
        # An empty paragraph is almost always an unset variable (`-m "$BODY"`), and an
        # empty subject makes a commit nothing can describe.
        if ! [[ "$1" =~ [^[:space:]] ]]; then
          echo "error: empty -m message" >&2
          exit 1
        fi
        # git itself separates -m paragraphs with a blank line. Without this the body
        # is glued onto the subject, so `git log --oneline` prints the whole message
        # as one line and every tool that reads a subject sees the entire commit.
        if [ "$FIRST_M" -eq 0 ]; then
          printf '\n' >> "$MSG_FILE"
        fi
        FIRST_M=0
        printf '%s\n' "$1" >> "$MSG_FILE"
        ;;
      -F)
        echo "error: -F cannot be combined with -m" >&2
        usage
        ;;
      *)
        usage
        ;;
    esac
    shift
  done
fi

# Whitespace-only counts as empty: `-F` of a blank file, or one that held nothing but
# the stripped trailer.
if ! grep -q '[^[:space:]]' "$MSG_FILE"; then
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
