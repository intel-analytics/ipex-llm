# Friesian Online Serving Helm Chart

This document demonstrates how to use the helm chart to deploy the Friesian online serving pipeline
on a Kubernetes cluster.

## Prepare environments

* AVX512 instruction support in nodes deploying the recall server
* [Preparation steps](../preparation) or a PV contains resource files
* [Helm](https://helm.sh) 3.2.0+
* Kubernetes 1.19+

## Modify the configurations

In [values.yaml](./values.yaml), you can edit parameters to match your cluster config.

Some important configs must be checked:

* Fill your resource PV name in [`resourcePVCName`](./values.yaml#L12).
* Check all resource paths and server config.
    * [`init.feature.resourcePath`](./values.yaml#L26)
    * [`init.feature.config`](./values.yaml#L31)
    * [`init.featureRecall.resourcePath`](./values.yaml#L51)
    * [`init.featureRecall.config`](./values.yaml#L56)
    * [`init.recall.resourcePath`](./values.yaml#L74)
    * [`init.recall.config`](./values.yaml#L81)
    * [`feature.config`](./values.yaml#L114)
    * [`featureRecall.config`](./values.yaml#L196)
    * [`recall.config`](./values.yaml#L275)
    * [`recall.resourcePath`](./values.yaml#L282)
    * [`ranking.config`](./values.yaml#L330)
    * [`ranking.resourcePath`](./values.yaml#L339)
    * [`recommender.config`](./values.yaml#L385)

* Modify all image repositories & tags to match your docker-hub.
    * [`init.image`](./values.yaml#L14)
    * [`feature.image`](./values.yaml#L89)
    * [`featureRecall.image`](./values.yaml#L171)
    * [`recall.image`](./values.yaml#L254)
    * [`ranking.image`](./values.yaml#L308)
    * [`recommender.image`](./values.yaml#L363)

* Check [all components](https://github.com/intel-analytics/BigDL/tree/main/scala/friesian) enabled
  settings are all you need.

| Component                      | Description                                                                                                                                                                                                                                                                                                                                                              | Switch                                 |
|--------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------|
| `init`                         | Initialize `redis` and `recall` components with the supplied resources                                                                                                                                                                                                                                                                                                   | `init.enabled`                         |
| `redis`                        | [Redis server](https://github.com/bitnami/charts/tree/master/bitnami/redis/) that `feature` and `featureRecall` depend on. <br />Can be disabled when external redis deployed. <br /> Nested components refer to [redis chart readme](https://github.com/bitnami/charts/tree/master/bitnami/redis/)                                                                      | `redis.enabled`                        |
| `prometheus`                   | [Prometheus stack](https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack) that serviceMonitors depend on. <br />Can be disabled when external prometheus stack is deployed. <br /> Nested components refer to [prometheus chart readme](https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack) | `prometheus.enabled`                   |
| `feature`                      | Feature server supply features                                                                                                                                                                                                                                                                                                                                           | Always On                              |
| `feature.serviceMonitor`       | Service Monitor for `feature`. <br />**`prometheus` must be enabled (or installed)**                                                                                                                                                                                                                                                                                     | `feature.serviceMonitor.enabled`       |
| `featureRecall`                | Feature server supply user embedding                                                                                                                                                                                                                                                                                                                                     | Always On                              |
| `featureRecall.serviceMonitor` | Service Monitor for `featureRecall`. <br />**`prometheus` must be enabled (or installed)**                                                                                                                                                                                                                                                                               | `featureRecall.serviceMonitor.enabled` |
| `recall`                       | Recall server                                                                                                                                                                                                                                                                                                                                                            | Always On                              |
| `recall.serviceMonitor`        | Service Monitor for `recall`. <br />**`prometheus` must be enabled (or installed)**                                                                                                                                                                                                                                                                                      | `recall.serviceMonitor.enabled`        |
| `ranking`                      | Ranking server                                                                                                                                                                                                                                                                                                                                                           | Always On                              |
| `ranking.serviceMonitor`       | Service Monitor for `ranking`. <br />**`prometheus` must be enabled (or installed)**                                                                                                                                                                                                                                                                                     | `ranking.serviceMonitor.enabled`       |
| `recommender`                  | Recommender http server                                                                                                                                                                                                                                                                                                                                                  | Always On                              |
| `recommender.serviceMonitor`   | Service Monitor for `recommender`. <br />**`prometheus` must be enabled (or installed)**                                                                                                                                                                                                                                                                                 | `recommender.serviceMonitor.enabled`   |
| `recommender.ingress`          | [Ingress](https://kubernetes.io/docs/concepts/services-networking/ingress/) for `recommender`. <br />**An [ingress controller](https://kubernetes.io/docs/concepts/services-networking/ingress-controllers/) must be installed**                                                                                                                                         | `recommender.ingress.enabled`          |

## Deploy the Friesian serving pipeline

To install the chart with the release name `my-release` in namespace `friesian`:

```bash
helm upgrade --install --debug -n friesian my-release ./friesian-helm --create-namespace
```

After installation, follow Helm output to check if Friesian serving works properly

## Cleanup

To uninstall/delete the `my-release` deployment:

```bash
helm delete my-release -n friesian
```

The Helm chart
uses [kube-prometheus-stack](https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack)
to monitor cluster.
This stack creates a Service for scraping kubelet information (defined
in `kube-prometheus-stack.prometheusOperator.kubeletService`) which isn't deleted when uninstalling.
See [this issue](https://github.com/SumoLogic/sumologic-kubernetes-collection/issues/1101) for a
detailed explanation.

To remove this service after uninstalling, run:

```bash
kubectl delete svc <release_name>-prometheus-kubelet -n kube-system

# In this demo, replace <release_name> with my-release
kubectl delete svc my-release-prometheus-kubelet -n kube-system
```

The commands remove all the Kubernetes components associated with the chart and delete the release.
