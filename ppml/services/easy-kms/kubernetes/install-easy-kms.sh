# Configure the variables to be passed into the templates.
export imageName=intelanalytics/easy-kms:2.3.0-SNAPSHOT # if enable sgx, replace with custom image
export dataStoragePath=a_host_path_for_persistent_stoarge
export serviceIP=your_key_management_service_ip_to_expose
export rootKey=your_256bit_base64_AES_key_string
export sgxEnabled=true

if [ ! -d $dataStoragePath ]
then
  echo "please create folder $dataStoragePath at host first"
  exit -1
fi

envsubst < easy-kms.yaml | kubectl apply -f -
