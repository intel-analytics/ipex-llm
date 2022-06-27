# Use Envoy to Scale out gRPC Services

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

We need to specify listeners and clusters as static_resources, and admin to manage the envoy service.

* admin: Optional. Module to manage envoy.
* resources:
    * listeners: Modules to listen requests and distribute requests to clusters.
    * clusters: Clusters to provide service.

Here is a sample envoy config file for ranking service:

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
      common_lb_config: { healthy_panic_thereshould: {value: 10} }
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
          unhealthy_threshold: 2
          healthy_threshold: 1
          grpc_health_check:
            service_name: ""
admin:
  address:
    socket_address: { address: 0.0.0.0, port_value: 9090 }
```

You need to modify the following properties according to your settings：

- `static_resources.listeners.name`: The unique name by which this listener is known. If no name is provided, Envoy will allocate an internal UUID for the listener.
- `static_resources.listeners.address.socket_address`: The address that the listener should listen on. In general, the address must be unique, though that is governed by the bind rules of the OS.
- `static_resources.listeners.filter_chains.filters.typed_config.route_config.virtual_hosts.routes.route.cluster`: Indicates the upstream cluster to which the request should be routed to. It Needs to be the same as `clusters.name`.
- `clusters.name`: Supplies the name of the cluster which must be unique across all clusters.
- `clusters.lb_policy`: The load balancer type to use when picking a host in the cluster.
- `clusters.common_lb_config.healthy_panic_threshold`: Configures the [healthy panic threshold](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/upstream/load_balancing/panic_threshold#arch-overview-load-balancing-panic-threshold). If not specified, the default is 50%. To disable panic mode, set to 0%.
- `clusters.load_assignment.cluster_name`: Name of the cluster.
- `clusters.load_assignment.endpoints`: List of endpoints to load balance to. Users need to add new `lb_endpoints` according to gRPC service listening to.
- `clusters.health_checks`: Optional [active health checking](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/upstream/health_checking#arch-overview-health-checking) configuration for the cluster. If no configuration is specified no health checking will be done and all cluster members will be considered healthy at all times.
- `admin`: The admin message is required to enable and configure the administration server. The Envoy admin endpoint can expose private information about the running service, allows modification of runtime settings and can also be used to shut the server down.


After modifying the config file, you can validate it with [validating-your-envoy-configuration](https://www.envoyproxy.io/docs/envoy/latest/start/quick-start/run-envoy#validating-your-envoy-configuration).

* here are the sample envoy config file for each service:
    * Inference: [inference_envoy.yaml](inference_envoy.yaml)
    * Feature Service: [feature_envoy.yaml](feature_envoy.yaml)
    * Feature Service - 2 tower: [feature_2tower_envoy.yaml](feature_2tower_envoy.yaml)
    * Vector Search: [vector.yaml](vector_envoy.yaml)
    * Recommend Server: [rec_ser_envoy.yaml](rec_ser_envoy.yaml)

## Run envoy

1. [download](https://www.envoyproxy.io/docs/envoy/latest/start/install) envoy `docker pull envoyproxy/envoy-dev:0dbd1fc9d972b991a1a3c2b8f638e2d40f134e44`
2. run command: `docker run --rm -it -p 8083:8083 -p 9090:9090 envoyproxy/envoy-dev:0dbd1fc9d972b991a1a3c2b8f638e2d40f134e44 --config-yaml "$(cat path/to/service-specific-envoy.yaml)" --parent-shutdown-time-s 1000000`
    **Note**: `-p 8083:8083 -p 9090:9090` need to be the same as `static_resources.listeners.address.socket_address.port_value` and `admin.address.socket_address.port_value`. If the admin is not set, you only need to publish the `static_resources.listeners.address.socket_address.port_value` here.
3. validate: run `netstat -tnlp` to see if the envoy process is listening to the corresponding port in the envoy config file.

## Scale out the gRPC Services

Use the Ranking service as an example:

* deployment: deploy [Ranking Service](online-serving.md#start-ranking-service) on server with IP 172.168.3.114 and  172.168.3.115, port=7083.
* choose one of the servers, 172.168.3.114 for example(or another new machine), to deploy the envoy process (as mentioned before) with [config file](#config-file) on port 8083. The port to listen is determined by the port assigned to Ranking Service in the gRPC config file.
* You can test if the Ranking Service is scaled and available by running the whole system.
