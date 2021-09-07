#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export NANO_HOME=${ANALYTICS_ZOO_ROOT}/python/nano/src
export NANO_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/nano/test

mypy --config-file ${NANO_TEST_DIR}/mypy.ini ${ANALYTICS_ZOO_ROOT}/python/nano
