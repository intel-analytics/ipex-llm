mvn -f ./scala clean test -Dsuites=com.intel.analytics.bigdl.dllib.optim.OptimPredictorShutdownSpec -DhdfsMaster=${hdfs_272_master} -P integration-test -DforkMode=never
