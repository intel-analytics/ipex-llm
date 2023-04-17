# delete service and deployment from k8s API server
if [ "$teeMode" = "sgx" ]; then
  kubectl delete -f bigdl-kms-deployment-sgx.yaml
elif [ "$teeMode" = "tdx" ]; then
  kubectl delete -f bigdl-kms-deployment-tdx.yaml
else
  echo "Wrong teeMode $teeMode! Expected sgx or tdx"
fi
