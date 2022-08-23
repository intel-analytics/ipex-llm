PYTHON_ZIP=$(find lib -name *-python-api.zip)
JAR=$(find lib -name *-jar-with-dependencies.jar)
export PYTHONPATH=$PYTHONPATH:$(pwd)/$PYTHON_ZIP
export PYTHONPATH=$PYTHONPATH:$(pwd)/$PYTHON_ZIP/bigdl/ppml/fl/nn/generated
export BIGDL_CLASSPATH=$JAR