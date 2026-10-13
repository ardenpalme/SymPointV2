#!/usr/bin/env bash
# PostToolUse: lint the Python file Claude just edited. Exit 2 sends ruff's output back to Claude.
f=$(jq -r '.tool_input.file_path // empty')
case "$f" in
  *.py) ruff check --force-exclude "$f" >&2 || exit 2 ;;
esac
exit 0
