## Envoy:

### Introduction:
1. Envoy is an L7 proxy and communication bus.
2. It's a self contained process that is designed to run alongside every application server. All of the Envoys form a transparent communication mesh in which each application sends and receives messages to and from localhost.
3. Written in C++ but independent of language. Run with the config file.[here is envoy.yaml]
4. Support HTTP/1.1, HTTP/2, HTTP/3 (after 1.19.0) and gRPC. Support load balancing.

### Config File:
* admin: Module to manage envoy
* resources:
    * listeners: Modules to listen requests and distribute requests to clusters.
    * clusters: Clusters to provide service.
* here are the sample envoy config file for each service:
    * Inference: [inference_envoy.yaml](src/main/resources/envoy/inference_envoy.yaml)
    * Feature Service: [feature_envoy.yaml](src/main/resources/envoy/feature_envoy.yaml)
    * Feature Service - 2 tower: [feature_2tower_envoy.yaml](src/main/resources/envoy/feature_2tower_envoy.yaml)
    * Vector Search: [vector.yaml](src/main/resources/envoy/vector_envoy.yaml)
    * Recommend Server: [rec_ser_envoy.yaml](src/main/resources/envoy/rec_ser_envoy.yaml)

### How to run envoy:
1. [download](https://www.envoyproxy.io/docs/envoy/latest/start/install) and deploy envoy(below use docker as example):
    * download: `docker pull envoyproxy/envoy-dev:21df5e8676a0f705709f0b3ed90fc2dbbd63cfc5`
2. run command: `docker run --rm -it  -p 9082:9082 -p 9090:9090 envoyproxy/envoy-dev:79ade4aebd02cf15bd934d6d58e90aa03ef6909e --config-yaml "$(cat path/to/service-specific-envoy.yaml)" --parent-shutdown-time-s 1000000`
3. validate: run `netstat -tnlp` to see if the envoy process is listening to the corresponding port in the envoy config file.

### Sample Procedure:
Use the Inference service as an example:

* deployment: run `Inference Service` on machine with IP 172.168.3.109 and  172.168.3.110, port=7073. For details on starting service, see [grpc](../README.md).
* choose one of the machine, 172.168.3.109 for example, to deploy the envoy process (as mentioned before) with [config file](inference_envoy.yaml) on port 8083. The port to listen is determined by the port assigned to Inference Service in the grpc config file.
* You can test if the Inference Service is scaled and available by running the whole system.

