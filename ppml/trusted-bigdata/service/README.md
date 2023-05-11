```bash
kubectl create secret generic kubeconfig-secret --from-file=<path_to_kubeconfig_file> -n bigdl-ppml
kubectl apply -f <path_to_password_folder>/password.yaml -n bigdl-ppml
kubectl apply -f <path_to_keys_folder>/keys.yaml -n bigdl-ppml
helm install trusted-bigdata-service .
kubectl get all -n bigdl-ppml # check if service pod is running
# if you want to access jupyter lab from frontend, forward port and access http://127.0.0.1:12345/?token=1234qwer from browser
# uninstall command: helm uninstall trusted-bigdata-service
```
