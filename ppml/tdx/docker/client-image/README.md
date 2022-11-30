## Build bigdl-tdx image
```bash
docker build \
  --build-arg http_proxy=.. \
  --build-arg https_proxy=.. \
  --build-arg no_proxy=.. \
  --rm --no-cache -t intelanalytics/bigdl-tdx-client-spark-3.1.3:2.2.0-SNAPSHOT .
```
