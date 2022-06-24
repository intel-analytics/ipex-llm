# Orca Known Issues

## **Estimator Issues**

### **OSError: Unable to load libhdfs: ./libhdfs.so: cannot open shared object file: No such file or directory**

This error occurs while running Orca with `yarn-client` mode on Cloudera, where PyArrow failed to locate `libhdfs.so` in default path of `$HADOOP_HOME/lib/native`. To solve this, we need to set the path of `libhdfs.so` in Cloudera to the environment variable of `ARROW_LIBHDFS_DIR` on spark executors. 

You could follow below steps:

1. use `locate libhdfs.so` to find `libhdfs.so`
2. `export ARROW_LIBHDFS_DIR=/opt/cloudera/parcels/CDH-5.15.2-1.cdh5.15.2.p0.3/lib64` (replace with the result of locate libhdfs.so)
3. If you are using `init_orca_context(cluster_mode="yarn-client")`: 
   ```
   conf = {"spark.executorEnv.ARROW_LIBHDFS_DIR": "/opt/cloudera/parcels/CDH-5.15.2-1.cdh5.15.2.p0.3/lib64"}
   init_orca_context(cluster_mode="yarn", conf=conf)
   ```
   If you are using `init_orca_context(cluster_mode="spark-submit")`:
   ```
   spark-submit --conf "spark.executorEnv.ARROW_LIBHDFS_DIR=/opt/cloudera/parcels/CDH-5.15.2-1.cdh5.15.2.p0.3/lib64"
   ```