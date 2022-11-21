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
export kmsIP=your_kms_ip_to_use_as
export dkeyserverNodeName=the_fixed_node_you_want_to_assign_dkeyserver_to #kubectl get nodes, and choose one

# Create k8s namespace and apply BigDL-eHSM-KMS
kubectl create namespace bigdl-ehsm-kms
kubectl label nodes $dkeyserverNodeName dkeyservernode=true
envsubst < bigdl-ehsm-kms.yaml | kubectl apply -f -
