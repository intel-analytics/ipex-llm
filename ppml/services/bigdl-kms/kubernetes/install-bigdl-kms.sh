# Configure the variables to be passed into the templates.
export imageName=intelanalytics/bigdl-kms-reference:2.5.0-SNAPSHOT # or replace with custom image of user
export dataStoragePath=a_host_path_for_persistent_stoarge
export serviceIP=your_key_management_service_ip_to_expose
export rootKey=your_256bit_base64_AES_key_string
export teeMode=sgx # sgx or tdx

if [ ! -d $dataStoragePath ]
then
  echo "please create folder $dataStoragePath at host first"
  exit -1
fi

if [ "$teeMode" = "sgx" ]; then
  envsubst < bigdl-kms-deployment-sgx.yaml | kubectl apply -f -
elif [ "$teeMode" = "tdx" ]; then
  envsubst < bigdl-kms-deployment-tdx.yaml | kubectl apply -f -
else
  echo "Wrong teeMode $teeMode! Expected sgx or tdx"
fi
