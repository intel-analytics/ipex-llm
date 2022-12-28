kubectl delete svc keywhiz-service -n bkeywhiz
kubectl delete svc bkeywhiz-frontend-service -n bkeywhiz
kubectl delete deployment bkeywhiz-frontend -n bkeywhiz
kubectl delete statefulsets keywhiz -n bkeywhiz
kubectl delete pvc mysql-persitent-storage-keywhiz-0 -n bkeywhiz
kubectl delete namespace bkeywhiz
kubectl delete pv mysql-nfs-pv
