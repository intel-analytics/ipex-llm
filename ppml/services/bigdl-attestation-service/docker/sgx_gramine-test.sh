set -ex
export ATTESTATION_SERVICE_HOST=your_attestation_service_host
export ATTESTATION_SERVICE_PORT=your_attestation_service_port
export APP_ID=your_app_id
export API_KEY=your_api_key

init_raw=`./init.sh | grep mr`
export MR_ENCLAVE=$(echo $init_raw | cut -c 13-77)
export MR_SIGNER=$(echo $init_raw | cut -c 89-153)

register_policy_resp=`java -Xmx1g -cp $BIGDL_HOME/jars/*:$SPARK_HOME/jars/*:$SPARK_HOME/examples/jars/*: com.intel.analytics.bigdl.ppml.attestation.admin.RegisterMrEnclave -u $ATTESTATION_SERVICE_HOST:$ATTESTATION_SERVICE_PORT -t BigDLRemoteAttestationService -i $APP_ID -k $API_KEY -e $MR_ENCLAVE -s $MR_SIGNER`
export POLICY_ID="${register_policy_resp##* }"

export sgx_command="java -Xmx1g -cp $BIGDL_HOME/jars/*:$SPARK_HOME/jars/*:$SPARK_HOME/examples/jars/*: com.intel.analytics.bigdl.ppml.attestation.AttestationCLI -u $ATTESTATION_SERVICE_HOST:$ATTESTATION_SERVICE_PORT -t BigDLRemoteAttestationService -i $APP_ID -k $API_KEY -o $POLICY_ID"
gramine-sgx bash | tee test.log