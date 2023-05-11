```bash
kubectl create secret generic kubeconfig-secret --from-file=<path_to_kubeconfig_file> -n bigdl-ppml
kubectl apply -f <path_to_password_folder>/password.yaml -n bigdl-ppml
kubectl apply -f <path_to_keys_folder>/keys.yaml -n bigdl-ppml
helm install trusted-bigdata-service .
# uninstall command: helm uninstall trusted-bigdata-service
```
