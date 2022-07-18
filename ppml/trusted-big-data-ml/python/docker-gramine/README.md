# Gramine dockerfile and related tests
## Build Gramine Docker image

Pull [BigDL](https://github.com/intel-analytics/BigDL). In `ppml/trusted-big-data-ml/python/docker-graphene`, replace the original `Dockerfile`, `Makefile` and `bash.manifest.template` with the files here. 

In `build-docker-image.sh`, change the output name to `intelanalytics/bigdl-ppml-trusted-big-data-ml-python-gramine:2.1.0-SNAPSHOT`, then run the script. 

## Build Jar for testing

In `gramine-examples`, run
  ```bash
  mvn package
  ```
## Start Gramine container
``` bash
# Provide your own path to the jar here
export PATH_TO_GRAMINE_EXAMPLES_JAR=narwhal/docker/spark-gramine/gramine-examples/target/gramine-examples-1.0-SNAPSHOT.jar

sudo docker run -itd \
    --net=host \
    --device=/dev/sgx/enclave \
    --device=/dev/sgx/provision \
    -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
    -v ${PATH_TO_GRAMINE_EXAMPLES_JAR}:/ppml/trusted-big-data-ml/gramine-examples-1.0-SNAPSHOT.jar  \
    --name=gramine-test \
    intelanalytics/bigdl-ppml-trusted-big-data-ml-python-gramine:2.1.0-SNAPSHOT bash

docker exec -it gramine-test bash
```

## Run the examples

In the container, run
``` bash
gramine-argv-serializer bash -c "java -cp /ppml/trusted-big-data-ml/gramine-examples-1.0-SNAPSHOT.jar com.intel.analytics.bigdl.ppml.FileSystemIO" > secured_argvs

gramine-sgx bash 2>&1 | tee test-gramine-FileSystemIO.log
```
``` bash
gramine-argv-serializer bash -c "java -cp /ppml/trusted-big-data-ml/gramine-examples-1.0-SNAPSHOT.jar com.intel.analytics.bigdl.ppml.ProcessBuilderTest" > secured_argvs

gramine-sgx bash 2>&1 | tee test-gramine-ProcessBuilderTest.log
```
