# Trusted Big Data ML with Occlum for production
This director is for production, and build occlum runable instance first.

You can see building command in [manually_build.yaml](https://github.com/intel-analytics/BigDL/blob/main/.github/workflows/manually_build.yml#L485) : bigdl-ppml-trusted-big-data-ml-scala-occlum-production.
It will build image normally, and then run occlum-build to build occlum runable instance (by running occlum init and build first) in /opt/occlum_spark. The default configuration is:
```bash
-e SGX_MEM_SIZE=20GB \
-e SGX_THREAD=2048 \
-e SGX_HEAP=1GB \
-e SGX_KERNEL_HEAP=1GB \
```

The image of the intermediate process is image_name=intelanalytics/bigdl-ppml-trusted-big-data-ml-scala-occlum-production:${TAG}, the final image is
final_name=intelanalytics/bigdl-ppml-trusted-big-data-ml-scala-occlum-production:${TAG}-build. But the final image size is too large because there are too many dependencies in this image.