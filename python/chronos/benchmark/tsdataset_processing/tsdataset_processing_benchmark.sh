set -e

bash $ANALYTICS_ZOO_ROOT/python/chronos/dev/release/release.sh linux default false
bash $ANALYTICS_ZOO_ROOT/python/nano/dev/build_and_install.sh linux default false pytorch --force-reinstall
whl_name=`ls $ANALYTICS_ZOO_ROOT/python/chronos/src/dist/`
pip install $ANALYTICS_ZOO_ROOT/python/chronos/src/dist/${whl_name}

cd $ANALYTICS_ZOO_ROOT/python/chronos/benchmark/tsdataset_processing/

echo "Chronos_Perf: Running TSDataset Processing Baseline"
source bigdl-nano-init
python tsdataset_processing.py --name "TSDataset Processing Baseline on nyc_taxi"
source bigdl-nano-unset-env
