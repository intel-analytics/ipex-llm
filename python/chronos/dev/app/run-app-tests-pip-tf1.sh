#!/usr/bin/env bash

clear_up () {
    echo "Clearing up environment. Uninstalling bigdl"
    pip uninstall -y bigdl-chronos
    pip uninstall -y bigdl-orca
    pip uninstall -y bigdl-dllib
    pip uninstall -y pyspark
}

chmod +x ${BIGDL_ROOT}/python/chronos/dev/app/ipynb2py.sh

set -e

echo "#1 start app test for chronos-network-traffic-autots-forecasting-deprecated"
#timer
start=$(date "+%s")
${BIGDL_ROOT}/python/chronos/dev/app/ipynb2py.sh ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/network_traffic_autots_forecasting_deprecated

FILENAME="${BIGDL_ROOT}/python/chronos/use-case/network_traffic/data/data.csv"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading network traffic data"

   wget $FTP_URI/analytics-zoo-data/network-traffic/data/data.csv -P ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/data

   echo "Finished downloading network traffic data"
fi

sed -i '/get_ipython()/d; /plot[.]/d; /plt[.]/d; /axs[.]/d' ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/network_traffic_autots_forecasting_deprecated.py
sed -i 's/epochs=2/epochs=1/g; s/object_store_memory=\"1/object_store_memory=\"10/' ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/network_traffic_autots_forecasting_deprecated.py
cd ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/

python ${BIGDL_ROOT}/python/chronos/use-case/network_traffic/network_traffic_autots_forecasting_deprecated.py
cd -

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "chronos network-traffic-autots-forecasting-deprecated failed"
    exit $exit_status
fi
now=$(date "+%s")
time1=$((now-start))
echo "#1 chronos-network-traffic-autots-forecasting-deprecated time used:$time1 seconds"

# This should be done at the very end after all tests finish.
clear_up
