kubectl delete svc couchdb -n bigdl-ehsm-kms
kubectl delete svc bigdl-ehsm-kms-service -n bigdl-ehsm-kms
kubectl delete svc dkeyserver -n bigdl-ehsm-kms
kubectl delete deployment bigdl-ehsm-kms-deployment -n bigdl-ehsm-kms
kubectl delete deployment dkeycache -n bigdl-ehsm-kms
kubectl delete statefulsets.apps couchdb -n bigdl-ehsm-kms
kubectl delete statefulsets.apps dkeyserver -n bigdl-ehsm-kms
kubectl delete pvc couch-persistent-storage-couchdb-0 -n bigdl-ehsm-kms
kubectl delete pvc domain-key-persistent-storage-dkeyserver-0
kubectl delete namespace bigdl-ehsm-kms
