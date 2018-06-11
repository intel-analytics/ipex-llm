# Check environment variables
if [ -z "${ANALYTICS_ZOO_HOME}" ]; then
    echo "Please set ANALYTICS_ZOO_HOME environment variable"
    exit 1
fi

export ANALYTICS_ZOO_JAR=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*jar-with-dependencies.jar"`

SPARK_OPTS='--master=local[*] --jars ${ANALYTICS_ZOO_JAR},./fraud-1.0.1-SNAPSHOT.jar --driver-memory 10g --executor-memory 10g' TOREE_OPTS='--nosparkcontext' jupyter notebook