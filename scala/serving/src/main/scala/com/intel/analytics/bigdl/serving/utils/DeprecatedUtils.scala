/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.serving.utils

import java.io.{File, FileInputStream}
import java.util.LinkedHashMap

import com.intel.analytics.bigdl.serving.ClusterServing
import org.yaml.snakeyaml.Yaml

object DeprecatedUtils {
  type HM = LinkedHashMap[String, String]
  /**
   * Deprecated load config method, only used in some existed tests
   */
  @deprecated
  def loadConfig(clusterServingHelper: ClusterServingHelper): Unit = {

    val yamlParser = new Yaml()
    val input = new FileInputStream(new File(clusterServingHelper.configPath))

    val configList = yamlParser.load(input).asInstanceOf[HM]

    // parse model field
    val modelConfig = configList.get("model").asInstanceOf[HM]
    clusterServingHelper.modelPath = if (clusterServingHelper.modelPath == "") {
      getYaml(modelConfig, "path", null).asInstanceOf[String]
    } else {
      clusterServingHelper.modelPath
    }
    clusterServingHelper.jobName = getYaml(modelConfig,
      "name", Conventions.SERVING_STREAM_DEFAULT_NAME).asInstanceOf[String]

    /**
     * Tensorflow usually use NHWC input
     * While others use NCHW
     */
    // parse data field
    val dataConfig = configList.get("data").asInstanceOf[HM]
    val redis = getYaml(dataConfig, "src", "localhost:6379").asInstanceOf[String]
    require(redis.split(":").length == 2, "Your redis host " +
      "and port are not valid, please check.")
    clusterServingHelper.redisHost = redis.split(":").head.trim
    clusterServingHelper.redisPort = redis.split(":").last.trim.toInt

    val secureConfig = configList.get("secure").asInstanceOf[HM]
    clusterServingHelper.redisSecureEnabled = getYaml(
      secureConfig, "secure_enabled", false).asInstanceOf[Boolean]

    val defaultPath = try {
      getClass.getClassLoader.getResource("keys/keystore.jks").getPath
    } catch {
      case _ => ""
    }
    clusterServingHelper.redisSecureTrustStorePath = getYaml(
      secureConfig, "secure_trust_store_path", defaultPath)
      .asInstanceOf[String]
    clusterServingHelper.redisSecureTrustStorePassword = getYaml(
      secureConfig, "secure_struct_store_password", "1234qwer").asInstanceOf[String]
    clusterServingHelper.modelEncrypted = getYaml(
      secureConfig, "model_encrypted", false).asInstanceOf[Boolean]
    clusterServingHelper.recordEncrypted = getYaml(
      secureConfig, "record_encrypted", false).asInstanceOf[Boolean]

    val typeStr = getYaml(dataConfig, "type", "image")
    require(typeStr != null, "data type in config must be specified.")

    clusterServingHelper.postProcessing = getYaml(dataConfig, "filter", "").asInstanceOf[String]
    clusterServingHelper.resize = getYaml(dataConfig, "resize", true).asInstanceOf[Boolean]
    clusterServingHelper.inputAlreadyBatched = getYaml(
      dataConfig, "batched", false).asInstanceOf[Boolean]

    val paramsConfig = configList.get("params").asInstanceOf[HM]
    clusterServingHelper.threadPerModel = getYaml(
      paramsConfig, "core_number", 4).asInstanceOf[Int]

    clusterServingHelper.modelParallelism = getYaml(
      paramsConfig, "model_number", default = 1).asInstanceOf[Int]

    clusterServingHelper.parseModelType(clusterServingHelper.modelPath)
    if (clusterServingHelper.modelType == "caffe" || clusterServingHelper.modelType == "bigdl") {
      if (System.getProperty("bigdl.engineType", "mklblas")
        .toLowerCase() == "mklblas") {
        clusterServingHelper.blasFlag = true
      }
      else clusterServingHelper.blasFlag = false
    }
    else clusterServingHelper.blasFlag = false

    val redisConfig = configList.get("redis").asInstanceOf[HM]
    clusterServingHelper.redisTimeout = getYaml(redisConfig, "timeout", 5000).asInstanceOf[Int]

  }
  /**
   * The util of getting parameter from yaml
   * @param configList the hashmap of this field in yaml
   * @param key the key of target field
   * @param default default value used when the field is empty
   * @return
   */
  def getYaml(configList: HM, key: String, default: Any): Any = {
    val configValue: Any = try {
      configList.get(key)
    } catch {
      case _ => null
    }
    if (configValue == null) {
      if (default == null) throw new Error(configList.toString + key + " must be provided")
      else {
        return default
      }
    }
    else {
      ClusterServing.logger.debug(s"Config list ${configList.toString} " +
        s"get key $key, value $configValue")
      configValue
    }
  }
}
