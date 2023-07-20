Replace the values in values.yaml
```bash
kubectl create namespace bigdl-ppml
kubectl create secret generic kubeconfig-secret --from-file=<path_to_kubeconfig_file> -n bigdl-ppml
kubectl apply -f <path_to_password_folder>/password.yaml -n bigdl-ppml
kubectl apply -f <path_to_keys_folder>/keys.yaml -n bigdl-ppml
helm install trusted-bigdata-service .
kubectl get all -n bigdl-ppml # check if service pod is running
# if you want to access jupyter lab from frontend, forward port and access http://notebook_pod_ip:jupyterPort/?token=1234qwer from browser
# uninstall command: helm uninstall trusted-bigdata-service
```
