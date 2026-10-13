#!/usr/bin/env bash
# Stop: don't let Claude finish while changed or new Python files fail ruff.
input=$(cat)
# Avoid an endless loop: if this hook already blocked once, let Claude stop.
[ "$(printf '%s' "$input" | jq -r '.stop_hook_active // false')" = "true" ] && exit 0
files=$( { git diff --name-only --diff-filter=d HEAD -- '*.py'
           git ls-files --others --exclude-standard -- '*.py'; } 2>/dev/null | sort -u )
[ -z "$files" ] && exit 0
printf '%s\n' "$files" | xargs -d '\n' ruff check --force-exclude >&2 || exit 2
exit 0
