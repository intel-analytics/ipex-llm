# Configure the variables to be passed into the templates.
export nfsServerIp=your_nfs_server_ip
export nfsPath=a_nfs_shared_folder_path_on_the_server
export couchdbRootUsername=YWRtaW4= # Reset is optional
export couchdbRootPassword=cGFzc3dvcmQ= # Reset is optional
# Set the versions according to your images
export dkeyserverImageName=intelccc/ehsm_dkeyserver:0.3.0
export couchdbImageName=couchdb:3.2
export dkeycacheImageName=intelccc/ehsm_dkeycache:0.3.0
export ehsmKmsImageName=intelccc/ehsm_kms:0.3.0
export pccsIP=your_pccs_IP
export dkeyserverIP=your_dkeyserver_ip_to_use_as
export kmsIP=your_kms_ip_to_use_as

# Create k8s namespace and apply BigDL-eHSM-KMS
kubectl create namespace bigdl-ehsm-kms
envsubst < bigdl-ehsm-kms.yaml | kubectl apply -f -