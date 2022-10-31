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

echo SGX_MEM_SIZE:$SGX_MEM_SIZE
echo SGX_LOG_LEVEL:$SGX_LOG_LEVEL
echo ATTESTATION:$ATTESTATION
if [[ "$ATTESTATION" == "false" ]]; then
   sed -i 's/"dcap"/"none"/g' bash.manifest.template
fi
make SGX=1 DEBUG=1 THIS_DIR=/ppml/trusted-big-data-ml  SPARK_USER=root G_SGX_SIZE=$SGX_MEM_SIZE G_LOG_LEVEL=$SGX_LOG_LEVEL
