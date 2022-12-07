echo $APP_ID $API_KEY $ATTESTATION_URL $MR_ENCLAVE $MR_SIGNER
JARS="$SPARK_HOME/jars/*:$SPARK_HOME/examples/jars/*:$BIGDL_HOME/jars/*"
java -cp $JARS com.intel.analytics.bigdl.ppml.attestation.admin.RegisterMrEnclave \
        -i $APP_ID -k $API_KEY -u $ATTESTATION_URL -s $MR_SIGNER -e $MR_ENCLAVE
