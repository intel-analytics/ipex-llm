#To add occlum mount conf and mkdir target dir
edit_json="$(cat Occlum.json | jq '.mount+=[{"target": "/opt/spark/pod-template","type": "hostfs","source": "/opt/spark/pod-template-copy"}]')" && \
        echo "${edit_json}" > Occlum.json
edit_json="$(cat Occlum.json | jq '.mount+=[{"target": "/var/run/secrets/kubernetes.io/serviceaccount","type": "hostfs","source": "/var/run/secrets/kubernetes.io/serviceaccount-copy"}]')" && \
        echo "${edit_json}" > Occlum.json
edit_json="$(cat Occlum.json | jq '.mount+=[{"target": "/opt/spark/conf","type": "hostfs","source": "/opt/spark/conf-copy"}]')" && \
        echo "${edit_json}" > Occlum.json

#mkdir target (should exist in occlum build)
mkdir -p /opt/spark/conf
mkdir -p /opt/spark/pod-template
mkdir -p /var/run/secrets/kubernetes.io/serviceaccount

#mkdir source ..data
mkdir -p /opt/spark/conf/..data
mkdir -p /opt/spark/pod-template/..data
mkdir -p /var/run/secrets/kubernetes.io/serviceaccount/..data

#make ..data not empty

#create test.txt
[[ -c ./test.txt ]] || touch ./test.txt
echo "test" > test.txt

# cp test.txt to ..data
cp ./test.txt /opt/spark/conf/..data/
cp ./test.txt /opt/spark/pod-template/..data/
cp ./test.txt /var/run/secrets/kubernetes.io/serviceaccount/..data/
rm ./test.txt
