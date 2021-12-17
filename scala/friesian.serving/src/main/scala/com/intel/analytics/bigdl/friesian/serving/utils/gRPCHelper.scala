/*
 * Copyright 2021 The BigDL Authors.
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

package com.intel.analytics.bigdl.friesian.serving.utils

import com.intel.analytics.bigdl.orca.inference.InferenceModel


import java.nio.file.Files
import scala.beans.BeanProperty
import org.apache.logging.log4j.{LogManager, Logger}

class gRPCHelper extends Serializable {
  // BeanProperty store attributes read from config file
  @BeanProperty var modelPath = ""
  @BeanProperty var ServicePort = 8980
  @BeanProperty var monitorPort: Int = 0

  // performance attributes
  @BeanProperty var inputAlreadyBatched = false
  @BeanProperty var coreNumberPerMachine = -1
  @BeanProperty var modelParallelism = 1
  @BeanProperty var threadPerModel = 1
  @BeanProperty var savedModelInputs: String = _

  // redis specific attributes
  @BeanProperty var redisUrl = "localhost:6379"
  @BeanProperty var redisMaxMemory = "4g"
  @BeanProperty var redisTimeout = 5000
  @BeanProperty var redisPoolMaxTotal = 256

  // feature service attributes
  @BeanProperty var serviceType = "kv"
  @BeanProperty var loadInitialData = false
  @BeanProperty var initialDataPath: String = _
  @BeanProperty var initialUserDataPath: String = _
  @BeanProperty var initialItemDataPath: String = _
  @BeanProperty var userFeatureColumns: String = _
  @BeanProperty var itemFeatureColumns: String = _
  @BeanProperty var userIDColumn: String = _
  @BeanProperty var itemIDColumn: String = _
  @BeanProperty var redisKeyPrefix: String = _
  @BeanProperty var redisClusterItemSlotType = 0

  // recall service attributes
  @BeanProperty var loadSavedIndex = false
  @BeanProperty var indexPath: String = _
  @BeanProperty var getFeatureFromFeatureService = false
  @BeanProperty var saveBuiltIndex = false
  @BeanProperty var indexDim: Int = 128
  @BeanProperty var part: Int = 20

  // feature & recall service attributes
  @BeanProperty var userModelPath: String = _
  @BeanProperty var itemModelPath: String = _

  // recommend service attributes
  @BeanProperty var inferenceColumns: String = _
  @BeanProperty var inferenceBatch: Int = 0
  @BeanProperty var recallServiceURL = "localhost:8980"
  @BeanProperty var featureServiceURL = "localhost:8980"
  @BeanProperty var rankingServiceURL = "localhost:8980"

  var configPath: String = "config.yaml"
  var redisHostPort: java.util.ArrayList[(String, Int)] = new java.util.ArrayList[(String, Int)]()
  var blasFlag: Boolean = false
  var savedModelInputsArr: Array[String] = _
  var userFeatureColArr: Array[String] = _
  var itemFeatureColArr: Array[String] = _
  var inferenceColArr: Array[String] = _
  var itemModel: InferenceModel = _
  var itemSlotType: Int = 0

  val logger: Logger = LogManager.getLogger(getClass)

  def parseConfigStrings(): Unit = {
    redisUrl.split("\\s*,\\s*").foreach(url => {
      val redisHost = url.split(":").head.trim
      val redisPort = url.split(":").last.trim.toInt
      redisHostPort.add(Tuple2(redisHost, redisPort))
    })
    if (redisKeyPrefix == null) {
      redisKeyPrefix = ""
    }
    // TODO: value check
    //    if (serviceType != "kv" && serviceType != "inference") {
    //      logger.error(s"serviceType must be 'kv' or 'inference' but got ${serviceType}")
    //    }
    if (savedModelInputs != null) {
      savedModelInputsArr = savedModelInputs.split("\\s*,\\s*")
    }

    if (userFeatureColumns != null) {
      userFeatureColArr = userFeatureColumns.split("\\s*,\\s*")
    }
    if (itemFeatureColumns != null) {
      itemFeatureColArr = itemFeatureColumns.split("\\s*,\\s*")
    }
    if (inferenceColumns != null) {
      inferenceColArr = inferenceColumns.split("\\s*,\\s*")
    }

    itemSlotType = if (redisClusterItemSlotType != 0 && redisClusterItemSlotType != 1 &&
      redisClusterItemSlotType != 2) {
      0
    } else {
      redisClusterItemSlotType
    }
  }

  /**
   * Load inference model
   * The concurrent number of inference model depends on
   * backend engine type
   * @return
   */
  def loadInferenceModel(concurrentNum: Int = 0, modelPathToLoad: String = modelPath,
                         savedModelInputs: Array[String] = null)
  : InferenceModel = {
    val model = new InferenceModel(modelParallelism)
    if (modelPathToLoad == "") {
      logger.error("The path to model should not be '', load model failed.");
    } else {
      val (modelType, defPath, weightPath) = parseModelType(modelPathToLoad)
      // Allow concurrent number overwrite
      if (concurrentNum > 0) {
        modelParallelism = concurrentNum
      }
      logger.info(
        s"gPRC load Inference Model with Parallelism $modelParallelism")

      modelType match {
        case "caffe" => model.doLoadCaffe(defPath, weightPath, blas = blasFlag)
        case "bigdl" => model.doLoadBigDL(weightPath, blas = blasFlag)
        case "tensorflowFrozenModel" =>
          model.doLoadTensorflow(weightPath, "frozenModel", 1, 1, true)
        case "tensorflowSavedModel" =>
          model.doLoadTensorflow(weightPath, "savedModel", savedModelInputs, null)
        case "pytorch" => model.doLoadPyTorch(weightPath)
        case "keras" => logger.error("Keras currently not supported in gRPC service," +
          "consider transform it to Tensorflow")
        case _ => logger.error("Invalid model type, please check your model directory")
      }
    }
    model
  }

  def setItemModel(itemModel: InferenceModel): Unit = {
    this.itemModel = itemModel
  }

  /**
   * Infer the model type in model directory
   * Try every file in the directory, infer which are the
   * model definition file and model weight file
   * @param location
   */
  def parseModelType(location: String): (String, String, String) = {
    /**
     * Download file to local if the scheme is remote
     * Currently support hdfs, s3
     */
    val scheme = location.split(":").head
    val localModelPath = if (scheme == "file" || location.split(":").length <= 1) {
      location.split("file://").last
    } else {
      val path = Files.createTempDirectory("model")
      val dstPath = path.getParent + "/" + path.getFileName
      Utils.copyToLocal(location, dstPath)
      dstPath
    }

    /**
     * Initialize all relevant parameters at first
     */
    var modelType:String = null
    var weightPath:String = null
    var defPath:String = null

    var variablesPathExist = false

    import java.io.File
    val f = new File(localModelPath)
    val fileList = f.listFiles

    if (fileList == null) {
      logger.error("Your model path provided in config is empty, please check your model path.")
    }
    // model type is always null, not support pass model type currently
    if (modelType == null) {

      for (file <- fileList) {
        val fName = file.getName
        val fPath = new File(localModelPath, fName).toString
        if (fName.endsWith("caffemodel")) {
          throwOneModelError(true, false, true, modelType, defPath, weightPath)
          weightPath = fPath
          modelType = "caffe"
        }
        else if (fName.endsWith("prototxt")) {
          throwOneModelError(false, true, false, modelType, defPath, weightPath)
          defPath = fPath
        }
        // ckpt seems not supported
        else if (fName.endsWith("pb")) {
          throwOneModelError(true, false, true, modelType, defPath, weightPath)
          weightPath = localModelPath
          if (variablesPathExist) {
            modelType = "tensorflowSavedModel"
          } else {
            modelType = "tensorflowFrozenModel"
          }
        }
        else if (fName.endsWith("pt")) {
          throwOneModelError(true, false, true, modelType, defPath, weightPath)
          weightPath = fPath
          modelType = "pytorch"
        }
        else if (fName.endsWith("model")) {
          throwOneModelError(true, false, true, modelType, defPath, weightPath)
          weightPath = fPath
          modelType = "bigdl"
        }
        else if (fName.endsWith("keras")) {
          throwOneModelError(true, false, true, modelType, defPath, weightPath)
          weightPath = fPath
          modelType = "keras"
        }
        else if (fName.endsWith("bin")) {
          throwOneModelError(true, false, true, modelType, defPath, weightPath)
          weightPath = fPath
          modelType = "openvino"
        }
        else if (fName.endsWith("xml")) {
          throwOneModelError(false, true, false, modelType, defPath, weightPath)
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
      if (modelType == null) logger.error("There is no model detected in your directory." +
        "Please refer to document for supported model types.")
    }
    else {
      modelType = modelType.toLowerCase
    }
    // auto set parallelism if coreNumberPerMachine is set
    if (coreNumberPerMachine > 0) {
      if (modelType == "openvino") {
        threadPerModel = coreNumberPerMachine
        modelParallelism = 1
      } else {
        threadPerModel = 1
        modelParallelism = coreNumberPerMachine
      }
    }

    (modelType, defPath, weightPath)
  }

  /**
   * To check if there already exists detected defPath or weightPath
   * @param defPath Boolean, true means need to check if it is not null
   * @param weightPath Boolean, true means need to check if it is not null
   */
  def throwOneModelError(modelType: Boolean,
                         defPath: Boolean, weightPath: Boolean, modelTypeStr: String,
                         defPathStr: String, weightPathStr: String)
  : Unit = {

    if ((modelType && modelTypeStr != null) ||
      (defPath && defPathStr != null) ||
      (weightPath && weightPathStr != null)) {
      logger.error("Only one model is allowed to exist in " +
        "model folder, please check your model folder to keep just" +
        "one model in the directory")

    }
  }
}
