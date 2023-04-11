## Build bigdl-tdx image
```bash
docker build \
  --build-arg http_proxy=.. \
  --build-arg https_proxy=.. \
  --build-arg no_proxy=.. \
  --build-arg BASE_IMAGE_TAG=.. \
  --rm --no-cache -t intelanalytics/bigdl-ppml-trusted-dl-serving:2.3.0-SNAPSHOT .
```
