---
name: clean-git-commits
description: >-
  Create git commits without Cursor co-author trailers so GitHub contributions
  stay on the human author only. Use when committing, amending, rebasing,
  pushing, or rewriting history in this repository.
---

# Clean Git Commits (no Cursor co-author)

## Rule

**Never** leave this in commit messages:

```text
Co-authored-by: Cursor <cursoragent@cursor.com>
```

Cursor injects that trailer when the agent runs `git commit` or `git commit --amend`. It attributes work to `cursoragent@cursor.com` on the GitHub contribution graph.

## Do not use

- `git commit`
- `git commit --amend`

Those commands re-add the Cursor trailer in this environment, even with `--no-verify`.

## Use instead

Run from repo root after staging (`git add`):

```bash
scripts/git_commit_clean.sh -m "Subject line" -m "Optional body paragraph."
```

Or with a message file:

```bash
scripts/git_commit_clean.sh -F /path/to/message.txt
```

## Rewrite an existing commit (remove Cursor trailer)

```bash
# Replace HEAD message; keeps tree and author date
MSG=$(mktemp)
git log -1 --format=%B | sed '/^Co-authored-by: Cursor/d' > "$MSG"
TREE=$(git rev-parse HEAD^{tree})
PARENT=$(git rev-parse HEAD^ 2>/dev/null || true)
if [ -n "$PARENT" ]; then
  NEW=$(git commit-tree "$TREE" -p "$PARENT" -F "$MSG")
else
  NEW=$(git commit-tree "$TREE" -F "$MSG")
fi
git reset --hard "$NEW"
rm "$MSG"
```

Force-push only when the user explicitly asks to update remote history.

## Verify before push

```bash
git log -1 --format=%B | grep -iE 'co-authored-by:.*cursor' && echo "FAIL" && exit 1
echo "OK"
```

## Prefer

When the user cares about attribution: stage changes and **ask them to commit**, or use `git_commit_clean.sh` only when they requested a commit.
