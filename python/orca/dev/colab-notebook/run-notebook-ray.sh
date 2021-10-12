#!/usr/bin/env bash
clear_up() {
	echo "Clearing up environment. Uninstalling analytics-zoo"
	pip uninstall -y bigdl-orca
	pip uninstall -y bigdl-dllib
	pip uninstall -y pyspark
}

set -e

runtime=0  # global variable that will be changed in run(); records temporary runtime

# the first argument is the number of ipynb, the second argument is the name of ipynb
run(){
	echo "#$1 start test for $2.ipynb"
	start=$(date "+%s")
	${ANALYTICS_ZOO_HOME}/python/orca/dev/colab-notebook/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/$2
	sed -i '/get_ipython/s/^/#/' ${ANALYTICS_ZOO_HOME}/$2.py
	python ${ANALYTICS_ZOO_HOME}/$2.py

	exit_status=$?
	if [ $exit_status -ne 0 ]; then
	  clear_up
	  echo "$2 failed"
	  exit $exit_status
	fi

	now=$(date "+%s")
	runtime=$((now - start))

	rm ${ANALYTICS_ZOO_HOME}/$2.py
}

# the first argument is the number of ipynb, the second argument is the name of ipynb,
# the third argument is the runtime used by this notebook
echo_time(){
	echo "#$1 $2 time used: $3 seconds"
}

name1="python/orca/colab-notebook/quickstart/ray_parameter_server"
run 1 $name1
runtime1=$runtime

echo_time 1 $name1 $runtime1
