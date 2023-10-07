## Build bigdl-trusted-bigdl-llm-fschat tdx image
```bash
docker build \
  --build-arg http_proxy=.. \
  --build-arg https_proxy=.. \
  --build-arg no_proxy=.. \
  --rm --no-cache -t intelanalytics/bigdl-ppml-trusted-fschat-tdx:2.4.0-SNAPSHOT .
```
