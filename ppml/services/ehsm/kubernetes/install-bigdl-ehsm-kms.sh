# Configure the variables to be passed into the templates.
export nfsServerIp=your_nfs_server_ip
export nfsPath=a_nfs_shared_folder_path_on_the_server
export couchdbRootUsername=YWRtaW4= # Reset is optional
export couchdbRootPassword=cGFzc3dvcmQ= # Reset is optional
# Set the versions according to your images
export pccsImageName=intelanalytics/pccs:0.3.0-SNAPSHOT
export dkeyserverImageName=intelccc/ehsm_dkeyserver:0.3.0
export couchdbImageName=couchdb:3.2
export dkeycacheImageName=intelccc/ehsm_dkeycache:0.3.0
export ehsmKmsImageName=intelccc/ehsm_kms:0.3.0
export pccsIP=IP=your_pccs_IP
export dkeyserverIP=your_dkeyserver_ip_to_use_as
export kmsIP=your_kms_ip_to_use_as
# Below are used by pccs
export apiKey=your_intel_pcs_server_subscription_key_obtained_through_web_registeration # Refer to README for obtaining pcs api key
# The below three service IPs are different IP addresses that unused in your subnetwork
# Please do not use the same real IPs as nodes
export countryName=your_country_name
export cityName=your_city_name
export organizaitonName=your_organizaition_name
export commonName=server_fqdn_or_your_name
export emailAddress=your_email_address
export serverPassword=your_server_password_to_use 

# Create k8s namespace and apply BigDL-eHSM-KMS
kubectl create namespace bigdl-ehsm-kms
envsubst < bigdl-ehsm-kms.yaml | kubectl apply -f -