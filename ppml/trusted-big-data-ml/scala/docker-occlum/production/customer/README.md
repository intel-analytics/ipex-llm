# Trusted Big Data ML with Occlum for customer
This director is for reducing occlum runable instance image size.
We assume the ${FINAL_NAME} is already build in ../production.

You can see building command in [manually_build.yaml](https://github.com/intel-analytics/BigDL/blob/main/.github/workflows/manually_build.yml#L526) : bigdl-ppml-trusted-big-data-ml-scala-occlum-production-customer.
It will build image by coping occlum runable instance (/opt/occlum_spark) and install necessary dependencies.

the final image is called intelanalytics/bigdl-ppml-trusted-big-data-ml-scala-occlum-production-customer:${TAG}.