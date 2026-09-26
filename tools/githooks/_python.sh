# Sourced by the hooks: finds a Python >= 3.10 and runs a check in tools/ with the given arguments.
run_tool() {
  root="$(git rev-parse --show-toplevel)"
  tool="$1"
  shift
  for py in python3 python "py -3"; do
    if $py -c "import sys; sys.exit(sys.version_info < (3, 10))" >/dev/null 2>&1; then
      $py "$root/tools/$tool" "$@"
      return $?
    fi
  done
  echo "no Python >= 3.10 found, so tools/$tool could not run" >&2
  return 1
}

# tools/check_tracked.py: local paths and binary data (AGENTS.md, hard rules 2 and 4).
run_check() {
  run_tool check_tracked.py "$@"
}
