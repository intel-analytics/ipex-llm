# Configure the variables to be passed into the templates.
export imageName=intelanalytics/easy-kms:2.3.0-SNAPSHOT
export dataStoragePath=a_host_path_for_persistent_stoarge
export serviceIP=your_key_management_service_ip_to_expose
export rootKey=your_256bit_base64_AES_key_string
export sgxEnabled=true

if [ ! -d $hostFolderPath ]
then
  echo "please create folder $hostFolderPath at host first"
  exit -1
fi

# Create k8s namespace and apply easy kms
kubectl create namespace easy-kms
envsubst < easy-kms.yaml | kubectl apply -f -
