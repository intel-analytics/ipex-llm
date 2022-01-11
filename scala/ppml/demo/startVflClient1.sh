docker run \
  --net=host \
  --device=/dev/sgx/enclave \
  --device=/dev/sgx/provision \
  -it intelanalytics/bigdl-ppml-trusted-big-data-fl-scala-graphene:flTest \
  bash /opt/runVflClient1.sh
