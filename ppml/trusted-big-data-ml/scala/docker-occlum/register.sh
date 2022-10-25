echo $APP_ID $API_KEY $ATTESTATION_URL $MR_ENCLAVE $MR_SIGNER
export FULL_URL="https://"$ATTESTATION_URL
java -cp MyTest-1.0-SNAPSHOT-jar-with-dependencies.jar python.Register 0 $APP_ID $API_KEY $FULL_URL $MR_ENCLAVE $MR_SIGNER
