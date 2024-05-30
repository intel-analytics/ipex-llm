#!/bin/bash

# wrapper for pytest command
# add this before `pytest ...` or `python -m pytest ...` to avoid unexpected exit code 127 caused by ipex on Windows
# ref: https://github.com/intel/intel-extension-for-pytorch/issues/634
pytest_check_error() {
  result=$(eval "$@" || echo "FINISH PYTEST")
  echo $result > pytest_check_error.log
  cat pytest_check_error.log
  failed_lines=$(cat pytest_check_error.log | { grep failed || true; })
  if [[ $failed_lines != "" ]]; then
    exit 1
  fi
  rm pytest_check_error.log
}

# wrapper for python command
# add this before `python ...` to avoid unexpected exit code 127 caused by ipex on Windows
# ref: https://github.com/intel/intel-extension-for-pytorch/issues/634
ipex_workaround_wrapper() {
    eval "$@" || ( [[ $? == 127 && $RUNNER_OS == "Windows" ]] && echo "EXIT CODE 127 DETECTED ON WINDOWS, IGNORE." || exit 1)
}
