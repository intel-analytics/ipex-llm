sgx_sign dump -enclave /opt/occlum_spark/build/lib/libocclum-libos.signed.so -dumpfile ../metadata_info_spark.txt
mr_enclave=$(sed -n -e '/enclave_hash.m/,/metadata->enclave_css.body.isv_prod_id/p' ../metadata_info_spark.txt |head -3|tail -2|xargs|sed 's/0x//g'|sed 's/ //g')
echo "mr_enclave" $mr_enclave
mr_signer=$(tail -2 ../metadata_info_spark.txt |xargs|sed 's/0x//g'|sed 's/ //g')
echo "mr_signer" $mr_signer
