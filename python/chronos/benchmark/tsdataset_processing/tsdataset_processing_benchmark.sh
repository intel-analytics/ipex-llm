set -e

bash $ANALYTICS_ZOO_ROOT/python/chronos/dev//release/release.sh linux default false
bash $ANALYTICS_ZOO_ROOT/python/nano/dev/build_and_install.sh linux default false pytorch
whl_name=`ls $ANALYTICS_ZOO_ROOT/python/chronos/dist/`
pip install $ANALYTICS_ZOO_ROOT/python/chronos/dist/${whl_name}

cd $ANALYTICS_ZOO_ROOT/python/chronos/benchmark/tsdataset_processing/

echo "Chronos_Perf: Running TSDataset Processing Baseline"
python tsdataset_processing.py --name "TSDataset Processing Baseline on nyc_taxi"
