#!/usr/bin/env bash

set -eu

usage() {
    echo "Usage: wait_for_server TIMEOUT IP PORT"
    exit 1
}

if [ $# -ne 3 ]; then
    usage
fi

exec timeout $1 bash -c "while ! nc -z -w1 $2 $3; do sleep 1; done"
