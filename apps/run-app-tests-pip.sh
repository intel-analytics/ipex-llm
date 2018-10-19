#!/usr/bin/env bash

export ANALYTICS_ZOO_HOME=${ANALYTICS_ZOO_ROOT}/dist

clear_up () {
    echo "Clearing up environment. Uninstalling analytics-zoo"
    pip uninstall -y analytics-zoo
    pip uninstall -y bigdl
    pip uninstall -y pyspark
}

chmod +x ${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh

echo "#1 start app test for anomaly-detection"
start=$(date "+%s")

# Conversion to py file and data preparation
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/anomaly-detection/anomaly-detection-nyc-taxi
chmod +x ${ANALYTICS_ZOO_HOME}/bin/data/NAB/nyc_taxi/get_nyc_taxi.sh
${ANALYTICS_ZOO_HOME}/bin/data/NAB/nyc_taxi/get_nyc_taxi.sh
sed "s/nb_epoch=20/nb_epoch=2/g; s/batch_size=1024/batch_size=1008/g" ${ANALYTICS_ZOO_HOME}/apps/anomaly-detection/anomaly-detection-nyc-taxi.py > ${ANALYTICS_ZOO_HOME}/apps/anomaly-detection/tmp_test.py

# Run the example
export SPARK_DRIVER_MEMORY=2g
python ${ANALYTICS_ZOO_HOME}/apps/anomaly-detection/tmp_test.py

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "anomaly-detection failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time1=$((now-start))
echo "anomaly-detection-nyc-taxi time used:$time1 seconds"

# This should be done at the very end after all tests finish.
clear_up
