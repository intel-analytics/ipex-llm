This is used for OSPDT review.

A separate Docker container layer tagged as: <YOUR_DOCKER_IMAGE>:<xPL_VERSION>-sources tag for sources of 3d party packages with MPL 1.x, MPL 2.x, GPL 1.x, GPL 2.x and GPL 3.x variants.

### Build Image
```bash
docker build \
  --build-arg http_proxy=.. \
  --build-arg https_proxy=.. \
  --build-arg no_proxy=.. \
  --rm --no-cache -t intelanalytics/ipex-llm:sources .
```
