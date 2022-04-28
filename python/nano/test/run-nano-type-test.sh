#!/bin/bash

export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export NANO_HOME=${ANALYTICS_ZOO_ROOT}/python/nano/
export NANO_TEST_DIR=${ANALYTICS_ZOO_ROOT}/python/nano/test
# "mypy --install-types --non-interactive" is to automatically install missing types 
mypy --install-types --non-interactive --config-file ${NANO_TEST_DIR}/mypy.ini $NANO_HOME/src $NANO_HOME/example
