#!/bin/bash
docker build \
  --build-arg http_proxy=http://proxy.iil.intel.com:911 \
  --build-arg https_proxy=http://proxy.iil.intel.com:911 \
  --build-arg no_proxy=localhost,127.0.0.1 \
  --rm --no-cache -t intelanalytics/ipex-llm-serving-xpu:2.1.0-SNAPSHOT-TEST .
