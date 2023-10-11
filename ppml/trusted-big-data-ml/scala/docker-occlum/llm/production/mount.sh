
#To avoid soft-link error, need to mkdir copy dir
[[ -d /var/run/secrets/kubernetes.io/serviceaccount/..data ]] || mkdir -p  /var/run/secrets/kubernetes.io/serviceaccount/..data
[[ -d /var/run/secrets/kubernetes.io/serviceaccount-copy ]] || mkdir -p /var/run/secrets/kubernetes.io/serviceaccount-copy
cp /var/run/secrets/kubernetes.io/serviceaccount/..data/* /var/run/secrets/kubernetes.io/serviceaccount-copy
