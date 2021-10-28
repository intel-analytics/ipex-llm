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


package com.intel.analytics.bigdl.serving

import com.intel.analytics.bigdl.orca.inference.InferenceModel
import com.intel.analytics.bigdl.serving.utils.Conventions
import redis.clients.jedis.Jedis

import scala.beans.BeanProperty

/**
 * The helper of Cluster Serving
 * by default, all parameters are loaded by config including model directory
 * However, in some condition, models are distributed to remote machine
 * and locate in tmp directory, but other configs are still needed.
 * Thus model directory could be passed and overwrite that in config YAML
 */
class ClusterServingHelper
  extends Serializable {
  // BeanProperty store attributes read from config file
  @BeanProperty var modelPath = ""
  @BeanProperty var jobName = Conventions.SERVING_STREAM_DEFAULT_NAME
  @BeanProperty var postProcessing = ""

  // utils attributes
  @BeanProperty var imageResize = ""

  // performance attributes
  @BeanProperty var inputAlreadyBatched = false
  @BeanProperty var coreNumberPerMachine = -1
  @BeanProperty var modelParallelism = 1
  @BeanProperty var threadPerModel = 1

  // specific attributes
  @BeanProperty var flinkRestUrl = "localhost:8081"
  @BeanProperty var queueUsed = "redis"
  @BeanProperty var kafkaUrl = "localhost:9092"
  @BeanProperty var redisUrl = "localhost:6379"
  @BeanProperty var redisMaxMemory = "4g"
  @BeanProperty var redisTimeout = 5000

  // secure attributes
  @BeanProperty var redisSecureEnabled = false
  @BeanProperty var redisSecureTrustStorePath = ""
  @BeanProperty var redisSecureTrustStorePassword = ""
  @BeanProperty var modelEncrypted = false
  @BeanProperty var recordEncrypted = false

  var configPath: String = "config.yaml"
  var redisHost: String = _
  var redisPort: Int = _
  var blasFlag: Boolean = false
  var chwFlag: Boolean = true
  var resize: Boolean = false
  var modelType: String = _
  var weightPath: String = _
  var defPath: String = _

  def parseConfigStrings(): Unit = {
    redisHost = redisUrl.split(":").head.trim
    redisPort = redisUrl.split(":").last.trim.toInt
  }
  /**
   * Load inference model
   * The concurrent number of inference model depends on
   * backend engine type
   * @return
   */
  def loadInferenceModel(concurrentNum: Int = 0): InferenceModel = {
    if (modelPath != "") {
      parseModelType(modelPath)
    }
    if (modelType.startsWith("tensorflow")) {
      chwFlag = false
    }
    // Allow concurrent number overwrite
    if (concurrentNum > 0) {
      modelParallelism = concurrentNum
    }
    ClusterServing.logger.info(
      s"Cluster Serving load Inference Model with Parallelism $modelParallelism")
    val model = new InferenceModel(modelParallelism)

    // Used for Tensorflow Model, it could not have intraThreadNum > 2^8
    // in some models, thus intraThreadNum should be limited

    var secret: String = null
    var salt: String = null
    if (modelEncrypted) {
      val jedis = new Jedis(redisHost, redisPort)
      while (secret == null || salt == null) {
        secret = jedis.hget(Conventions.MODEL_SECURED_KEY, Conventions.MODEL_SECURED_SECRET)
        salt = jedis.hget(Conventions.MODEL_SECURED_KEY, Conventions.MODEL_SECURED_SALT)
        ClusterServing.logger.info("Waiting for Model Encrypted Secret and Salt in Redis," +
          "please put them in model_secured -> secret and " +
          "model_secured -> salt")
        ClusterServing.logger.info("Retrying in 3 seconds...")
        Thread.sleep(3000)
      }

    }
    modelType match {
      case "caffe" => model.doLoadCaffe(defPath, weightPath, blas = blasFlag)
      case "bigdl" => model.doLoadBigDL(weightPath, blas = blasFlag)
      case "tensorflowFrozenModel" =>
        model.doLoadTensorflow(weightPath, "frozenModel", 1, 1, true)
      case "tensorflowSavedModel" =>
        model.doLoadTensorflow(weightPath, "savedModel", null, null)
      case "pytorch" => model.doLoadPyTorch(weightPath)
      case "keras" => logError("Keras currently not supported in Cluster Serving," +
        "consider transform it to Tensorflow")
      case "openvino" => modelEncrypted match {
        case true => model.doLoadEncryptedOpenVINO(
          defPath, weightPath, secret, salt, threadPerModel)
        case false => model.doLoadOpenVINONg(defPath, weightPath, threadPerModel)
      }
      case _ => logError("Invalid model type, please check your model directory")
    }
    model
  }

  /**
   * To check if there already exists detected defPath or weightPath
   * @param defPath Boolean, true means need to check if it is not null
   * @param weightPath Boolean, true means need to check if it is not null
   */
  def throwOneModelError(modelType: Boolean,
                         defPath: Boolean, weightPath: Boolean): Unit = {

    if ((modelType && this.modelType != null) ||
        (defPath && this.defPath != null) ||
        (weightPath && this.weightPath != null)) {
      logError("Only one model is allowed to exist in " +
        "model folder, please check your model folder to keep just" +
        "one model in the directory")

    }
  }

  /**
   * Log error message to local log file
   * @param msg
   */
  def logError(msg: String): Unit = {
    ClusterServing.logger.error(msg)
    throw new Error(msg)
  }


  /**
   * Infer the model type in model directory
   * Try every file in the directory, infer which are the
   * model definition file and model weight file
   * @param location
   */
  def parseModelType(location: String): Unit = {
    /**
     * Download file to local if the scheme is remote
     * Currently support hdfs, s3
     */
    val scheme = location.split(":").head
    val localModelPath = if (scheme == "file" || location.split(":").length <= 1) {
      location.split("file://").last
    } else {
      // hdfs and s3 are supported by Flink natively
      null
    }

    /**
     * Initialize all relevant parameters at first
     */
    modelType = null
    weightPath = null
    defPath = null

    var variablesPathExist = false

    import java.io.File
    val f = new File(localModelPath)
    val fileList = f.listFiles

    if (fileList == null) {
      logError("Your model path provided in config is empty, please check your model path.")
    }
    // model type is always null, not support pass model type currently
    if (modelType == null) {

      for (file <- fileList) {
        val fName = file.getName
        val fPath = new File(localModelPath, fName).toString
        if (fName.endsWith("caffemodel")) {
          throwOneModelError(true, false, true)
          weightPath = fPath
          modelType = "caffe"
        }
        else if (fName.endsWith("prototxt")) {
          throwOneModelError(false, true, false)
          defPath = fPath
        }
        // ckpt seems not supported
        else if (fName.endsWith("pb")) {
          throwOneModelError(true, false, true)
          weightPath = localModelPath
          if (variablesPathExist) {
            modelType = "tensorflowSavedModel"
          } else {
            modelType = "tensorflowFrozenModel"
          }
        }
        else if (fName.endsWith("pt")) {
          throwOneModelError(true, false, true)
          weightPath = fPath
          modelType = "pytorch"
        }
        else if (fName.endsWith("model")) {
          throwOneModelError(true, false, true)
          weightPath = fPath
          modelType = "bigdl"
        }
        else if (fName.endsWith("keras")) {
          throwOneModelError(true, false, true)
          weightPath = fPath
          modelType = "keras"
        }
        else if (fName.endsWith("bin")) {
          throwOneModelError(true, false, true)
          weightPath = fPath
          modelType = "openvino"
        }
        else if (fName.endsWith("xml")) {
          throwOneModelError(false, true, false)
          defPath = fPath
        }
        else if (fName.equals("variables")) {
          if (modelType != null && modelType.equals("tensorflowFrozenModel")) {
            modelType = "tensorflowSavedModel"
          } else {
            variablesPathExist = true
          }
        }

      }
      if (modelType == null) logError("There is no model detected in your directory." +
        "Please refer to document for supported model types.")
    }
    else {
      modelType = modelType.toLowerCase
    }
    // auto set parallelism if coreNumberPerMachine is set
    if (coreNumberPerMachine > 0) {
      if (modelType == "openvino") {
        threadPerModel = coreNumberPerMachine
      } else {
        threadPerModel = 1
        modelParallelism = coreNumberPerMachine
      }
    }
  }

}
object ClusterServingHelper {
  /**
   * Method wrapped for external use only
   * @param modelDir directory of model
   * @param concurrentNumber model concurrent number
   * @return
   */
  def loadModelfromDir(modelDir: String, concurrentNumber: Int = 1): (InferenceModel, String) = {
    val helper = new ClusterServingHelper()
    helper.modelPath = modelDir
    helper.parseModelType(modelDir)
    (helper.loadInferenceModel(concurrentNumber), helper.modelType)
  }
}
