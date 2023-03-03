cd /ppml
./init.sh
export sgx_command="./tensorflow_model_server --model_base_path=path_to_model --model_name=name_of_model --rest_api_port=rest_port --tensorflow_inter_op_parallelism=2 --tensorflow_intra_op_parallelism=5"

gramine-sgx bash 2>&1 | tee tfserving-sgx.log

