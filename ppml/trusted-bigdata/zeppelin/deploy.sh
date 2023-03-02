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
APISERVER=https://172.168.0.205:6443
# like https://127.0.0.1:8000
TOKEN=eyJhbGciOiJSUzI1NiIsImtpZCI6Imc0YldfWmtmTlNyYWNybmJpazhLMTJQOVNKUk1TYThCb2tjTnFMNk1fSncifQ.eyJpc3MiOiJrdWJlcm5ldGVzL3NlcnZpY2VhY2NvdW50Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9uYW1lc3BhY2UiOiJkZWZhdWx0Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9zZWNyZXQubmFtZSI6InplcHBlbGluLXRva2VuLW5zNGh0Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9zZXJ2aWNlLWFjY291bnQubmFtZSI6InplcHBlbGluIiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9zZXJ2aWNlLWFjY291bnQudWlkIjoiNjM4M2QwNzctOTA5MC00OTRmLTg5MGItZDNmM2UwYmU1YzJlIiwic3ViIjoic3lzdGVtOnNlcnZpY2VhY2NvdW50OmRlZmF1bHQ6emVwcGVsaW4ifQ.fHpD5IDftlKMi_UCITUkAfKsoYrw9SzZpGZYj_nOki8tfs1iW6zQWss3OLbWRlxGI99x3ayRrsAxWTD4ONE_isyWe4bMlNr4BDraO-q44mi6iVZX_owobYI49M2pYX5HBvX9KrHyVdF3Y8ljw3_M3nqvw_4TFjBo9WWt22BnCQcGFxtiA8_O_sLj6E2gkX0yWrwD1nq5i2O4j8uYysv4QWrhv22wdxrL8fhzJbtUYpYpwPGmJlcq6VzWuz4IBchsmVfOEufRdxTGIphv8M6jzq9Fa4JR8mE4pnyhY-No9Bpw9m4BVEghxn2kABjjOkpMCbonPB5dfIxzwerjA58HeQ
set -x
#CLUSTER_NAME="kubernetes"
#APISERVER=$(kubectl config view -o jsonpath="{.clusters[?(@.name==\"$CLUSTER_NAME\")].cluster.server}")
#TOKEN=$(kubectl get secrets -o jsonpath="{.items[?(@.metadata.annotations['kubernetes\.io/service-account\.name']=='zeppelin')].data.token}"|base64 -d)

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
