## Build bigdl-tdx image
```bash
docker build \
  --build-arg http_proxy=.. \
  --build-arg https_proxy=.. \
  --build-arg no_proxy=.. \
  --rm --no-cache -t intelanalytics/bigdl-ppml-trusted-big-data:2.3.0-SNAPSHOT .
```
