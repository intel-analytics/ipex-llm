# Configure the variables to be passed into the templates.
export nfsServerIp=your_nfs_server_ip
export nfsPath=a_nfs_shared_folder_path_on_the_server
export mysqlImage=mysql:8
export keywhizServerImage=intelanalytics/keywhiz
export bigdlKmsFrontendImage=intelanalytics/keywhiz
export kmsIP=your_kms_ip_to_use_as

# Create k8s namespace and apply BigDL-KMS
kubectl create namespace bigdl-kms
envsubst < bigdl-kms.yaml | kubectl apply -f -
