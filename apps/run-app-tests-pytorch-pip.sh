#!/usr/bin/env bash

export ANALYTICS_ZOO_HOME=${ANALYTICS_ZOO_ROOT}

clear_up () {
    echo "Clearing up environment. Uninstalling analytics-zoo"
    pip uninstall -y bigdal-orca
    pip uninstall -y bigdl-dllib
}

chmod +x ${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh

set -e

echo "#start app test for pytorch face-generation"
#timer
start=$(date "+%s")
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/pytorch/face_generation
sed -i '/get_ipython()/d' ${ANALYTICS_ZOO_HOME}/apps/pytorch/face_generation.py
sed -i '/plt./d' ${ANALYTICS_ZOO_HOME}/apps/pytorch/face_generation.py
python ${ANALYTICS_ZOO_HOME}/apps/pytorch/face_generation.py
exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "pytorch face-generation failed"
    exit $exit_status
fi
now=$(date "+%s")
time=$((now-start))
echo "#pytorch face-generation time used:$time seconds"
