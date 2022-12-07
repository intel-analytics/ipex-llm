#!/bin/bash
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# echo commands to the terminal output
set -ex

# Check whether there is a passwd entry for the container UID
myuid=$(id -u)
mygid=$(id -g)
# turn off -e for getent because it will return error code in anonymous uid case
set +e
uidentry=$(getent passwd $myuid)
set -e

# Set PCCS conf
if [ "$PCCS_URL" != "" ] ; then
    echo 'PCCS_URL='${PCCS_URL}'/sgx/certification/v3/' > /etc/sgx_default_qcnl.conf
    echo 'USE_SECURE_CERT=FALSE' >> /etc/sgx_default_qcnl.conf
fi

# If there is no passwd entry for the container UID, attempt to create one
if [ -z "$uidentry" ] ; then
    if [ -w /etc/passwd ] ; then
        echo "$myuid:x:$myuid:$mygid:anonymous uid:$SPARK_HOME:/bin/false" >> /etc/passwd
    else
        echo "Container ENTRYPOINT failed to add passwd entry for anonymous UID"
    fi
fi


# We do not have any arguments, just run bash
if [ "$#" == 0 ]; then
  echo "[INFO] no command is passed in"
  echo "[INFO] enter pass-through mode"
  exec /usr/bin/tini -s -- "bash"
fi


# Attestation
if [ -z "$ATTESTATION" ]; then
  echo "[INFO] Attestation is disabled!"
  ATTESTATION="false"
fi

echo $SGX_ENABLED

runtime_command="$@"
if [ "$SGX_ENABLED" == "true" ]; then
  if [ "$ATTESTATION" ==  "true" ]; then 
    rm /ppml/temp_command_file || true
    bash attestation.sh
    if [ "$ENCRYPTED_FSD" == "true" ]; then
      echo "[INFO] Distributed encrypted file system is enabled"
      bash encrypted-fsd.sh
    fi
    echo $runtime_command >> temp_command_file
    export sgx_command="bash temp_command_file && rm temp_command_file"
  else 
    if [ "$ENCRYPTED_FSD" == "true" ]; then
      rm /ppml/temp_command_file || true
      echo "[INFO] Distributed encrypted file system is enabled"
      bash encrypted-fsd.sh
      echo $runtime_command >> temp_command_file
      export sgx_command="bash temp_command_file && rm temp_command_file"
    else
      export sgx_command=$runtime_command
    fi
  fi
  ./init.sh && \
  gramine-sgx bash 2>&1
  rm /ppml/temp_command_file || true
else
  exec $runtime_command
fi
