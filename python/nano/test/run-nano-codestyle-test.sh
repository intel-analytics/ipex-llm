#!/bin/bash

set -e

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export NANO_HOME=${ANALYTICS_ZOO_ROOT}/python/nano/src
export NANO_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/nano/test

pycodestyle ${NANO_HOME} --config=${NANO_TEST_DIR}/tox.ini

echo "coding style test pass"
pydocstyle --ignore D104,D100,D212,D203,D401,D402 ${NANO_HOME}/bigdl/nano/tf
pydocstyle --ignore D104,D100,D212,D203,D401,D402 ${NANO_HOME}/bigdl/nano/pytorch/*.py
pydocstyle --ignore D104,D100,D212,D203,D401,D402 ${NANO_HOME}/bigdl/nano/pytorch/optim/
pydocstyle --ignore D104,D100,D212,D203,D401,D402 ${NANO_HOME}/bigdl/nano/pytorch/plugins/
