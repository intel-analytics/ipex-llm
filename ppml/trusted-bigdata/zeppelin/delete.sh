#!/bin/bash

# You should set the required environment variable before curl command
#APISERVER="https://<kubernetes-api-server>"
#TOKEN="<access-token>"

# Delete the zeppelin-server-conf-map ConfigMap
curl -k -X DELETE \
  -H "Content-Type: application/yaml" \
  -H "Authorization: Bearer $TOKEN" \
  $APISERVER/api/v1/namespaces/default/configmaps/zeppelin-server-conf-map

# Delete the zeppelin-server-conf ConfigMap
curl -k -X DELETE \
  -H "Content-Type: application/yaml" \
  -H "Authorization: Bearer $TOKEN" \
  $APISERVER/api/v1/namespaces/default/configmaps/zeppelin-server-conf

# Delete the zeppelin-server Deployment
curl -k -X DELETE \
  -H "Content-Type: application/yaml" \
  -H "Authorization: Bearer $TOKEN" \
  $APISERVER/apis/apps/v1/namespaces/default/deployments/zeppelin-server

# Delete the zeppelin-server service
curl -k -X DELETE \
  -H "Content-Type: application/yaml" \
  -H "Authorization: Bearer $TOKEN" \
  $APISERVER/api/v1/namespaces/default/services/zeppelin-server

# Delete the zeppelin-server ServiceAccount
curl -k -X DELETE \
  -H "Content-Type: application/yaml" \
  -H "Authorization: Bearer $TOKEN" \
  $APISERVER/api/v1/namespaces/default/serviceaccounts/zeppelin-server

# Delete the zeppelin-server-role ClusterRole
curl -k -X DELETE \
  -H "Content-Type: application/yaml" \
  -H "Authorization: Bearer $TOKEN" \
  $APISERVER/apis/rbac.authorization.k8s.io/v1/clusterroles/zeppelin-server-role

# Delete the zeppelin-server-role-binding RoleBinding
curl -k -X DELETE \
  -H "Content-Type: application/yaml" \
  -H "Authorization: Bearer $TOKEN" \
  $APISERVER/apis/rbac.authorization.k8s.io/v1/namespaces/default/rolebindings/zeppelin-server-role-binding

