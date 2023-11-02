# JupyterLab in Gramine
This is a tutorial about how to run JupyterLab in Gramine

## What is JupyterLab?
JupyterLab is an advanced web-based interactive development environment for Jupyter notebooks, enabling data scientists and researchers to create, collaborate, and share documents containing live code, equations, visualizations, and explanatory text. It offers an integrated, extensible environment that supports multiple programming languages, code execution, interactive data visualization, and more.
For more information, please refer to [Jupyter official website](https://jupyter.org/).

## How to run JupyterLab in Gramine?
Please ensure that there is an available k8s cluster . And we use the [helm chart](https://helm.sh/) for deployment.

### Prepare parameters
Replace the values in [values.yaml](./values.yaml). The specific parameter meanings have been listed in [values.yaml](./values.yaml).

### Prepare environment
```bash
kubectl create namespace bigdl-ppml-jupyter
```
This will create a namespace called bigdl-ppml-jupyter to avoid naming conflict and organize resources.
```bash
kubectl create secret generic kubeconfig-secret --from-file=<path_to_kubeconfig_file> -n bigdl-ppml-jupyter
```
This will create secret containing k8s information.
```bash
kubectl apply -f <path_to_password_folder>/password.yaml -n bigdl-ppml-jupyter
kubectl apply -f <path_to_keys_folder>/keys.yaml -n bigdl-ppml-jupyter
```
This creates some security related resources. `password.yaml` and `keys.yaml` come from [generate script](https://github.com/intel-analytics/BigDL/tree/main/ppml/scripts).
### Install and use
```bash
helm install trusted-bigdata-service .
```
This will create JupterLab.
```bash
kubectl get all -n bigdl-ppml-jupyter
```
This will check if service pod is running. If you want to access jupyter lab from frontend, forward port and access http://notebookExternalIP:jupyterPort/?token=1234qwer from browser
### Uninstall
```bash
helm uninstall trusted-bigdata-service
```
