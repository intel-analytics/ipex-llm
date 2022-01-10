docker run \
  --net=host \
  --device=/dev/sgx_enclave \
  --device=/dev/sgx_provision \
  -it intelanalytics/bigdl-ppml-trusted-big-data-fl-scala-graphene:flTest \
  bash /opt/runFlserver.sh

