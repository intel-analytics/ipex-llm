```bash
helm install trusted-deep-learning-service .
kubectl create secret generic kubeconfig-secret --from-file=<path_to_kubeconfig_file>
kubectl get pods # check if service pod is running
# if you want to access jupyter lab from frontend, forward port and access http://127.0.0.1:12345/?token=1234qwer from browser
# uninstall command: helm uninstall trusted-deep-learning-service
```
