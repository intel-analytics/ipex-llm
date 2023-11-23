## Build bigdl-trusted-bigdl-llm tdx window-OS-base image
```bash
docker build \
  --build-arg http_proxy=.. \
  --build-arg https_proxy=.. \
  --build-arg no_proxy=.. \
  --rm \
  --no-cache \
  -t intelanalytics/bigdl-ppml-trusted-bigdl-llm-tdx-windows:2.5.0-SNAPSHOT \
  -f ./Dockerfile .
```
