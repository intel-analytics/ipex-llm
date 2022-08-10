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

# Prepare basic envs
export SGX_MEM_SIZE=$SGX_EXECUTOR_MEM_SIZE && \
export spark_commnd="/opt/jdk8/bin/java -Dlog4j.configurationFile=/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/log4j2.xml -Xms1G -Xmx$SGX_EXECUTOR_JVM_MEM_SIZE "${SPARK_EXECUTOR_JAVA_OPTS[@]}" -cp "$SPARK_CLASSPATH" org.apache.spark.executor.CoarseGrainedExecutorBackend --driver-url $SPARK_DRIVER_URL --executor-id $SPARK_EXECUTOR_ID --cores $SPARK_EXECUTOR_CORES --app-id $SPARK_APPLICATION_ID --hostname $SPARK_EXECUTOR_POD_IP --resourceProfileId $SPARK_RESOURCE_PROFILE_ID" && \
# Generate secured-argvs for attestation
if [ "$ATTESTATION" = "true" ]; then
  spark_commnd=$ATTESTATION_COMMAND" && "$spark_commnd
fi
echo $spark_commnd && \
/graphene/Tools/argv_serializer bash -c "export TF_MKL_ALLOC_MAX_BYTES=10737418240 && export _SPARK_AUTH_SECRET=$_SPARK_AUTH_SECRET && $spark_commnd" > /ppml/trusted-big-data-ml/secured-argvs
