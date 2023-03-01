#!/bin/bash

# Split the original YAML file into separate files based on the delimiter '---'
csplit -s -z -f "yaml-file-" zeppelin-server-deployment.yaml "/^---/" {*}

# Rename the created files with their corresponding resource types
mv yaml-file-00 ConfigMap1.yaml
mv yaml-file-01 ConfigMap2.yaml
mv yaml-file-02 Deployment.yaml
mv yaml-file-03 Service.yaml
mv yaml-file-04 ServiceAccount.yaml
mv yaml-file-05 ClusterRole.yaml
mv yaml-file-06 RoleBinding.yaml

# Set the required environment variables, you should uncomment it
#APISERVER= <YOUR_K8S_APISERVER>
# like https://127.0.0.1:8000
#TOKEN= <YOUR_ZEPPELIN_SERVICE_ACCOUNT_TOKEN>
set -x
CLUSTER_NAME="kubernetes"
APISERVER=$(kubectl config view -o jsonpath="{.clusters[?(@.name==\"$CLUSTER_NAME\")].cluster.server}")
TOKEN=$(kubectl get secrets -o jsonpath="{.items[?(@.metadata.annotations['kubernetes\.io/service-account\.name']=='zeppelin')].data.token}"|base64 -d)

# Deploy zeppelin-server-conf-map
curl -k -X POST \
  -H "Content-Type: application/yaml" \
  -H "Authorization: Bearer $TOKEN" \
  --data-binary @ConfigMap1.yaml \
  $APISERVER/api/v1/namespaces/default/configmaps

# Deploy zeppelin-server-conf
curl -k -X POST \
  -H "Content-Type: application/yaml" \
  -H "Authorization: Bearer $TOKEN" \
  --data-binary @ConfigMap2.yaml \
  $APISERVER/api/v1/namespaces/default/configmaps

# Deploy zeppelin-server
curl -k -X POST \
  -H "Content-Type: application/yaml" \
  -H "Authorization: Bearer $TOKEN" \
  --data-binary @Deployment.yaml \
  $APISERVER/apis/apps/v1/namespaces/default/deployments

# Deploy zeppelin-server service
curl -k -X POST \
  -H "Content-Type: application/yaml" \
  -H "Authorization: Bearer $TOKEN" \
  --data-binary @Service.yaml \
  $APISERVER/api/v1/namespaces/default/services

# Deploy zeppelin-server ServiceAccount
curl -k -X POST \
  -H "Content-Type: application/yaml" \
  -H "Authorization: Bearer $TOKEN" \
  --data-binary @ServiceAccount.yaml \
  $APISERVER/api/v1/namespaces/default/serviceaccounts

# Deploy zeppelin-server-role
curl -k -X POST \
  -H "Content-Type: application/yaml" \
  -H "Authorization: Bearer $TOKEN" \
  --data-binary @ClusterRole.yaml \
  $APISERVER/apis/rbac.authorization.k8s.io/v1/clusterroles

# Deploy zeppelin-server-role-binding
curl -k -X POST \
  -H "Content-Type: application/yaml" \
  -H "Authorization: Bearer $TOKEN" \
  --data-binary @RoleBinding.yaml \
  $APISERVER/apis/rbac.authorization.k8s.io/v1/namespaces/default/rolebindings


# Remove these yaml
rm ConfigMap1.yaml ConfigMap2.yaml Deployment.yaml Service.yaml ServiceAccount.yaml ClusterRole.yaml RoleBinding.yaml
