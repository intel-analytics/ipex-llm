# Trusted Big Data ML with Occlum for customer
This director is for reducing occlum runable instance image size.
We assume the ${FINAL_NAME} is already build in ../production.

You can see building command in [manually_build.yaml](https://github.com/intel-analytics/BigDL/blob/main/.github/workflows/manually_build.yml#L526) : bigdl-ppml-trusted-big-data-ml-scala-occlum-production-customer.
It will build image by coping occlum runable instance (/opt/occlum_spark) and install necessary dependencies.

the final image is called `intelanalytics/bigdl-ppml-trusted-big-data-ml-scala-occlum-production-customer:${TAG}`.


### Run application in docker and k8s

#### Docker
Set policy_Id to ENV.
```
export policy_Id=${policy_Id}
```
or
```bash
#start-spark-local.sh
-e ${policy_Id}
```

#### K8s
Add policy_Id to driver and executor ENV.
```yaml
#driver.yaml
env:
  - name: policy_Id
    value: "${policy_Id}"
```

```yaml
#executor.yaml
env:
  - name: policy_Id
    value: "${policy_Id}"
```