# Sourced by the hooks: finds a Python >= 3.10 and runs tools/check_tracked.py with the given arguments.
run_check() {
  root="$(git rev-parse --show-toplevel)"
  for py in python3 python "py -3"; do
    if $py -c "import sys; sys.exit(sys.version_info < (3, 10))" >/dev/null 2>&1; then
      $py "$root/tools/check_tracked.py" "$@"
      return $?
    fi
  done
  echo "no Python >= 3.10 found, so tools/check_tracked.py could not run" >&2
  return 1
}
