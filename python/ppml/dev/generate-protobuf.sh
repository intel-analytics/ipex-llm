# This script generate the python protobuf code from proto file in scala/ppml
# Prerequisite: pip install grpcio-tools
set -x
SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})
PROTO_PATH=$SCRIPT_DIR/../../../scala/ppml/src/main/proto
if [ ! -d "py_proto" ]; then
    mkdir py_proto
fi

python -m grpc_tools.protoc -I$PROTO_PATH --python_out=./py_proto --grpc_python_out=./py_proto $PROTO_PATH/*.proto