set -e

bash $ANALYTICS_ZOO_ROOT/python/chronos/dev/release/release.sh linux default false
bash $ANALYTICS_ZOO_ROOT/python/nano/dev/build_and_install.sh linux default false pytorch --force-reinstall
whl_name=`ls $ANALYTICS_ZOO_ROOT/python/chronos/src/dist/`
pip install $ANALYTICS_ZOO_ROOT/python/chronos/src/dist/${whl_name}[pytorch]

cd $ANALYTICS_ZOO_ROOT/python/chronos/benchmark/forecaster_training/

echo "Chronos_Perf: Running TCNForecaster Training Baseline"
source bigdl-nano-init
python forecaster_training.py --name "TCNForecaster Training Baseline"
source bigdl-nano-unset-env