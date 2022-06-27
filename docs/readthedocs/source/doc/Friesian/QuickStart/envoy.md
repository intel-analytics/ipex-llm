# Use Envoy to Scale gRPC Deployment

## Overview

Users can use [service proxy](online-serving.md#service-proxy) to scale out the gRPC services to multiple servers. Envoy can be used as the routing and load balancing substrate for gRPC requests and responses.

## Key Concepts

### Envoy
1. Envoy is an L7 proxy and communication bus.
2. It's a self-contained process that is designed to run alongside every application server. All the Envoys form a transparent communication mesh in which each application sends and receives messages to and from localhost.
3. Written in C++ but independent of language. Run with the config file.
4. Support HTTP/1.1, HTTP/2, HTTP/3 (after 1.19.0) and gRPC. Support load balancing.

## Config File

Envoy should be started with YAML config file. Here we use static configuration as an example.

We need to specify listeners and clusters as static_resources, and admin tu manage the envoy service.

* admin: Module to manage envoy
* resources:
    * listeners: Modules to listen requests and distribute requests to clusters.
    * clusters: Clusters to provide service.

Here is a sample envoy config for ranking service:

```yaml
static_resources:
  listeners:
    - name: ranking_listener
      address:
        socket_address: { address: 0.0.0.0, port_value: 8083}
      filter_chains:
        - filters:
            - name: envoy.filters.network.http_connection_manager
              typed_config:
                "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
                stat_prefix: ingress_http
                codec_type: AUTO
                route_config:
                  name: local_route
                  virtual_hosts:
                    - name: local_service
                      domains: ["*"]
                      routes:
                        - match: { prefix: "/" }
                          route: { cluster: ranking_service }
                http_filters:
                  - name: envoy.filters.http.router
                common_http_protocol_options: { idle_timeout: 0s }
                stream_idle_timeout: 0s
  clusters:
    - name: ranking_service
      connect_timeout: 5s
      lb_policy: ROUND_ROBIN
      type: STATIC
      http2_protocol_options: {}
      load_assignment:
        cluster_name: ranking_service
        endpoints:
            - lb_endpoints:
                - endpoint:
                    address:
                        socket_address:
                            address: 172.168.3.114
                            port_value: 7083
            - lb_endpoints:
                - endpoint:
                    address:
                        socket_address:
                            address: 172.168.3.115
                            port_value: 7083
      health_checks:
        - timeout: 1s
          interval: 60s
          interval_jitter: 1s
          unhealthy_threshold: 4
          healthy_threshold: 1
          grpc_health_check:
            service_name: ""
admin:
  address:
    socket_address: { address: 0.0.0.0, port_value: 9090 }
```

You need to modify the following properties according to your settings：


After modifying the config file, you can validate it with [validating-your-envoy-configuration](https://www.envoyproxy.io/docs/envoy/latest/start/quick-start/run-envoy#validating-your-envoy-configuration).

* here are the sample envoy config file for each service:
    * Inference: [inference_envoy.yaml](inference_envoy.yaml)
    * Feature Service: [feature_envoy.yaml](feature_envoy.yaml)
    * Feature Service - 2 tower: [feature_2tower_envoy.yaml](feature_2tower_envoy.yaml)
    * Vector Search: [vector.yaml](vector_envoy.yaml)
    * Recommend Server: [rec_ser_envoy.yaml](rec_ser_envoy.yaml)

## Run envoy
1. [download](https://www.envoyproxy.io/docs/envoy/latest/start/install) and deploy envoy(below use docker as example):
    * download: `docker pull envoyproxy/envoy-dev:21df5e8676a0f705709f0b3ed90fc2dbbd63cfc5`
    * deploy: `docker run -itd --name envoy_c --net=host envoyproxy/envoy-dev:79ade4aebd02cf15bd934d6d58e90aa03ef6909e`
    * login into container: `docker exec -it envoy_c bash`
2. run command: `envoy -c path/to/service-specific-envoy.yaml --parent-shutdown-time-s 1000000`
3. validate: run `netstat -tnlp` to see if the envoy process is listening to the corresponding port in the envoy config file.

## Sample Procedure
Use the Inference service as an example:

* deployment: run `Inference Service` on machine with IP 172.168.3.109 and  172.168.3.110, port=7073. For details on starting service, see [grpc](../README.md).
* choose one of the machine, 172.168.3.109 for example, to deploy the envoy process (as mentioned before) with [config file](inference_envoy.yaml) on port 8083. The port to listen is determined by the port assigned to Inference Service in the grpc config file.
* You can test if the Inference Service is scaled and available by running the whole system.


