#!/bin/bash

export ORCA_HOME=${BIGDL_HOME}/python/orca/
export ORCA_TEST_DIR=${BIGDL_HOME}/python/orca/test
# "mypy --install-types --non-interactive" is to automatically install missing types 
mypy --install-types --non-interactive --config-file ${ORCA_TEST_DIR}/mypy.ini $ORCA_HOME/src/bigdl/orca/data/utils.py $ORCA_HOME/src/bigdl/orca/data/ray_xshards.py
