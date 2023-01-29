#register application

#get mrenclave mrsigner
MR_ENCLAVE_temp=$(bash print_enclave_signer.sh | grep mr_enclave)
MR_ENCLAVE_temp_arr=(${MR_ENCLAVE_temp})
export MR_ENCLAVE=${MR_ENCLAVE_temp_arr[1]}
MR_SIGNER_temp=$(bash print_enclave_signer.sh | grep mr_signer)
MR_SIGNER_temp_arr=(${MR_SIGNER_temp})
export MR_SIGNER=${MR_SIGNER_temp_arr[1]}

#register and get policy_Id
policy_Id_temp=$(bash register.sh | grep policy_Id)
policy_Id_temp_arr=(${policy_Id_temp})
export policy_Id=${policy_Id_temp_arr[1]}
echo "policy_Id "${policy_Id}