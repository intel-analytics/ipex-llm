kubectl delete svc keywhiz-service -n bigdl-kms
kubectl delete svc bigdl-kms-frontend-service -n bigdl-kms
kubectl delete deployment bigdl-kms-frontend -n bigdl-kms
kubectl delete statefulsets keywhiz -n bigdl-kms
kubectl delete pvc mysql-persitent-storage-keywhiz-0 -n bigdl-kms
kubectl delete namespace bigdl-kms
kubectl delete pv mysql-nfs-pv
