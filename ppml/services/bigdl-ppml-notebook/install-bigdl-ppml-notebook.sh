# Configure the variables to be passed into the templates.
export imageName=bigdl-ppml-trusted-bigdata-gramine-reference:2.4.0-SNAPSHOT # or replace with custom image of user
export k8sMasterURL=k8s://$(sudo kubectl cluster-info | grep 'https.*6443' -o -m 1)
export teeMode=sgx # sgx or tdx

if [ "$teeMode" = "sgx" ]; then
  envsubst < bigdl-ppml-notebook-sgx.yaml | kubectl apply -f -
elif [ "$teeMode" = "tdx" ]; then
  envsubst < bigdl-ppml-notebook-tdx.yaml | kubectl apply -f -
else
  echo "Wrong teeMode $teeMode! Expected sgx or tdx"
fi
