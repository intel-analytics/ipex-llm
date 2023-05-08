# Configure the variables to be passed into the templates.
export nfsServerIp=your_nfs_server_ip
export nfsPath=a_nfs_shared_folder_path_on_the_server
export imageName=bigdl-ppml-trusted-bigdata-gramine-reference:2.4.0-SNAPSHOT # or replace with custom image of user
export TEEMode=sgx # sgx or tdx


export k8sMasterURL=k8s://$(sudo kubectl cluster-info | grep 'https.*6443' -o -m 1)

if [ "$TEEMode" = "sgx" ]; then
  envsubst < bigdl-ppml-notebook-sgx.yaml | kubectl apply -f -
elif [ "$TEEMode" = "tdx" ]; then
  envsubst < bigdl-ppml-notebook-tdx.yaml | kubectl apply -f -
else
  echo "Wrong teeMode $TEEMode! Expected sgx or tdx"
fi
