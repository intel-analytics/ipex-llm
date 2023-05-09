#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

class PPMLConf:
    def __init__(self, app_name, sgx_enabled = True):
        self.spark_conf = SparkConf().setAppName(app_name)
        self.init_spark_on_k8s_conf(sgx_enabled)
        return self

    def set(self, key, value):
        self.spark_conf.set(key, value)
    
    def conf(self):
        return self.spark_conf

    def init_spark_on_k8s_conf(self, sgx_enabled):
        master = os.getenv("RUNTIME_SPARK_MASTER")
        image = os.getenv("RUNTIME_K8S_SPARK_IMAGE")
        driver_ip = os.getenv("LOCAL_IP")
        print("k8s master url is " + str(master))
        print("executor image is " + str(image))
        print("driver ip is " + str(driver_ip))

        secure_password = os.getenv("secure_password")

        self.spark_conf.setMaster(master)
            .set("spark.submit.deployMode", "client")
            .set("spark.kubernetes.container.image", image)
            .set("spark.driver.host", driver_ip)
            .set("spark.kubernetes.driver.podTemplateFile", "/ppml/spark-driver-template.yaml")
            .set("spark.kubernetes.executor.podTemplateFile", "/ppml/spark-executor-template.yaml")
            .set("spark.kubernetes.authenticate.driver.serviceAccountName", "spark")
            .set("spark.kubernetes.executor.deleteOnTermination", "false")
            .set("spark.network.timeout", "10000000")
            .set("spark.executor.heartbeatInterval", "10000000")
            .set("spark.python.use.daemon", "false")
            .set("spark.python.worker.reuse", "false")
            .set("spark.authenticate", "true")
            .set("spark.authenticate.secret", secure_password)
            .set("spark.kubernetes.executor.secretKeyRef.SPARK_AUTHENTICATE_SECRET", "spark-secret:secret")
            .set("spark.kubernetes.driver.secretKeyRef.SPARK_AUTHENTICATE_SECRET", "spark-secret:secret")

        if sgx_enabled:
            self.spark_conf.set("spark.kubernetes.sgx.enabled", "true")
                .set("spark.kubernetes.sgx.driver.mem", "64g")
                .set("spark.kubernetes.sgx.driver.jvm.mem", "12g")
                .set("spark.kubernetes.sgx.executor.mem", "64g")
                .set("spark.kubernetes.sgx.executor.jvm.mem", "12g")
