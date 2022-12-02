#!/bin/bash

set -ex

export ORCA_HOME=${BIGDL_HOME}/python/orca
export ORCA_DEVTEST_DIR=${BIGDL_HOME}/python/orca/dev/test
export PYTHONPATH=${ORCA_HOME}/src:${PYTHONPATH}

# TODO: read from another file
# "mypy --install-types --non-interactive" is to automatically install missing types 
mypy --install-types --non-interactive --config-file ${ORCA_DEVTEST_DIR}/mypy.ini \
                                                     $ORCA_HOME/src/bigdl/orca/data/ \
                                                     $ORCA_HOME/src/bigdl/orca/ray/ \
                                                     $ORCA_HOME/src/bigdl/orca/inference/ \
                                                     $ORCA_HOME/src/bigdl/orca/learn/trigger.py \
                                                     $ORCA_HOME/src/bigdl/orca/learn/metrics.py
