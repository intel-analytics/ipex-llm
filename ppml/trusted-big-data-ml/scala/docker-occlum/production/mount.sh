
#To avoid soft-link error, need to mkdir copy dir
[[ -d /opt/spark/conf/..data ]] || mkdir /opt/spark/conf/..data
[[ -d /opt/spark/conf-copy ]] || mkdir /opt/spark/conf-copy
cp /opt/spark/conf/..data/* /opt/spark/conf-copy

[[ -d /opt/spark/pod-template/..data ]] || mkdir /opt/spark/pod-template/..data
[[ -d /opt/spark/pod-template-copy ]] || mkdir /opt/spark/pod-template-copy
cp /opt/spark/pod-template/..data/* /opt/spark/pod-template-copy

[[ -d /var/run/secrets/kubernetes.io/serviceaccount/..data ]] || mkdir -p  /var/run/secrets/kubernetes.io/serviceaccount/..data
[[ -d /var/run/secrets/kubernetes.io/serviceaccount-copy ]] || mkdir -p /var/run/secrets/kubernetes.io/serviceaccount-copy
cp /var/run/secrets/kubernetes.io/serviceaccount/..data/* /var/run/secrets/kubernetes.io/serviceaccount-copy
