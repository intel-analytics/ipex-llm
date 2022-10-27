echo $APP_ID $API_KEY $ATTESTATION_URL $MR_ENCLAVE $MR_SIGNER
export FULL_URL="https://"$ATTESTATION_URL
JARS="$SPARK_HOME/jars/*:$SPARK_HOME/examples/jars/*:$BIGDL_HOME/jars/*"
java -cp $JARS com.intel.analytics.bigdl.ppml.attestation.RegisterMrenclave $APP_ID $API_KEY $FULL_URL $MR_ENCLAVE $MR_SIGNER