# Trusted Realtime ML with occlum

Please pay attention to IP and path etc.. They should be changed to your own server IP/path.

## How To Build

Before running the following command, please modify the paths in `build-docker-image.sh` first. Then build docker image by running this command:

```bash
./build-docker-image.sh
```

## How To Run

### Prepare the keys

PPML in analytics zoo needs secured keys to enable Flink TLS, https and TLS enabled Redis. You need to prepare secure keys and keystores.

This script is under `analytics-zoo/ppml/scripts`:

```bash
../../../scripts/generate-keys.sh
```

You also need to store password you used in the previous step in a secured file:

This script is also under `/analytics-zoo/ppml/scripts`:

```bash
../../../scripts/generate-password.sh used_password_when_generate_keys
```

For example:

```bash
../../../scripts/generate-password.sh 1234qwer
```

### Start Trusted Cluster Serving with PPML Docker image

The default operating system limits on mmap counts is likely to be too low, which may result in out of memory exceptions.
To address this, run the following command before stating trusted cluster serving:
```bash
sudo sysctl -w vm.max_map_count=655300
```
This change will propagate into the containers as they share the same kernel as the host OS.

#### Local mode (Single container)

In this mode, all components, redis, Flink & http front end, are running in single container.

Before run the following command, please modify paths in `start-local-cluster-serving.sh` first. Then start Trusted Cluster Serving with the following command:

```bash
./start-local-cluster-serving.sh
```

##### Troubleshooting
You can run the script `/opt/check-status.sh` in the docker container to check whether the components have been correctly started.
Note that this only works for local cluster serving (for now).

To test a specific component, pass one or more argument to it among the following:
"redis", "flinkjm", "flinktm", "frontend", and "serving". For example, run the following command to check the status of the Flink job manager.

```bash
docker exec -it trusted-realtime-mllocal bash /opt/check-status.sh flinkjm
```

To test all components, you can either pass no argument or pass the "all" argument.

```bash
docker exec -it trusted-realtime-mllocal bash /opt/check-status.sh
```
If all is well, the following results should be displayed:

```
Detecting redis status...
Redis initialization successful.
Detecting Flink job manager status...
Flink job manager initialization successful.
Detecting Flink task manager status...
Flink task manager initialization successful.
Detecting http frontend status. This may take a while.
Http frontend initialization successful.
Detecting cluster-serving-job status...
cluster-serving-job initialization successful.
```

It is suggested to run this script once after starting local cluster serving to verify that all components are up and running.

#### Distributed mode (Multi-containers/Multi-nodes)

In this mode, all components, redis, Flink & http front end, are running in different containers. Some of them can be distributed to multi-nodes. 

Pre-requests:

1. Setup `no password ssh login` between all nodes.
2. Modify IP/paths in `environment.sh`. 

```bash
nano environment.sh
```

##### Start distributed cluster serving
To start all the services of distributed cluster serving, run
```bash
./start-distributed-cluster-serving.sh
```
You can also run the following command to start the flink jobmanager and taskmanager containers only:
```bash
./deploy-flink.sh
```
##### Stop distributed cluster serving 
To stop all the services of distributed cluster serving, run
```bash
./stop-distributed-cluster-serving.sh
```
You can also run the following command to stop the flink jobmanager and taskmanager only:
```bash
./stop-flink.sh
```
