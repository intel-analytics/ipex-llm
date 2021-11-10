#!/bin/bash

export NANO_HOME=${ANALYTICS_ZOO_ROOT}/python/nano/src
export NANO_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/nano/test

pycodestyle ${NANO_HOME} --config=${NANO_TEST_DIR}/tox.ini
