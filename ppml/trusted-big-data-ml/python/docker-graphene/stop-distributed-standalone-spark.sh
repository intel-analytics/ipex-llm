
#!/bin/bash

#set -x

source ./environment.sh

echo ">>> Standalone spark service"
ssh root@$MASTER "docker rm -f spark-driver"
