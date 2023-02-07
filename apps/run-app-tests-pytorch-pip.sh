#!/usr/bin/env bash

clear_up () {
    echo "Clearing up environment. Uninstalling BigDL"
    pip uninstall -y bigdl-orca
    pip uninstall -y bigdl-dllib
}

chmod +x ${BIGDL_ROOT}/apps/ipynb2py.sh

set -e

echo "#start app test for pytorch face-generation"
#timer
start=$(date "+%s")
${BIGDL_ROOT}/apps/ipynb2py.sh ${BIGDL_ROOT}/apps/pytorch/face_generation
sed -i '/get_ipython()/d' ${BIGDL_ROOT}/apps/pytorch/face_generation.py
sed -i '/plt./d' ${BIGDL_ROOT}/apps/pytorch/face_generation.py
python ${BIGDL_ROOT}/apps/pytorch/face_generation.py
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
