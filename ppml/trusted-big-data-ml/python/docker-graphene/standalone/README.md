# Spark Standalone Mode in SGX

[Spark Standalone](https://spark.apache.org/docs/latest/spark-standalone.html) is a simple deploy mode provided by Spark. With the help of Graphene-SGX, we are able to setup Spark Standalone (Spark Master and Spark Worker) within SGX, and deploy PPML applications upon them.

WARNING: Spark Standalone on SGX is not recommended anymore. Using it in production will encounter performance, resource management and deployment issues. Please refer to [Spark Kubernetes on SGX](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/python/docker-graphene/kubernetes/README.md).

**Modify variables and paths in `environment.sh`, e.g., `YOUR_LOCAL_ENCLAVE_KEY_PATH`.**

## Start the container to run spark applications in spark standalone mode

 Run the following commands

```bash
./deploy-distributed-standalone-spark.sh
./start-distributed-spark-driver.sh
```

Then use `distributed-check-status.sh` to check master's and worker's status and make sure that both of them are running.

Use the following commands to enter the docker of spark driver.

```bash
sudo docker exec -it spark-driver bash
cd /ppml/trusted-big-data-ml
./init.sh
./standalone/start-spark-standalone-driver-sgx.sh
```

## Run PySpark examples

To run the PySpark examples in spark standalone mode, you only need to replace the following command in spark local mode command:

```bash
--master 'local[4]' \
```

with

```bash
--master 'spark://your_master_url' \
--conf spark.authenticate=true \
--conf spark.authenticate.secret=your_secret_key \
```

and  replace `your_master_url` with your own master url and `your_secret_key` with your own secret key.
