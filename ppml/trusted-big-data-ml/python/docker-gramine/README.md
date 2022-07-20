# Gramine dockerfile and related tests
## Build Gramine Docker image

Pull [BigDL](https://github.com/intel-analytics/BigDL). In `ppml/trusted-big-data-ml/python/docker-graphene`, replace the original `Dockerfile`, `Makefile` and `bash.manifest.template` with the files here. 

In `build-docker-image.sh`, change the output name to `intelanalytics/bigdl-ppml-trusted-big-data-ml-python-gramine:2.1.0-SNAPSHOT`, then run the script. 