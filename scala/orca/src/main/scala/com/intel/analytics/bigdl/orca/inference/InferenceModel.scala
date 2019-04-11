/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.pipeline.inference

import java.lang.{Float => JFloat, Integer => JInt}
import java.util
import java.util.concurrent.LinkedBlockingQueue
import java.util.{List => JList}

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.zoo.pipeline.inference.DeviceType.DeviceTypeEnumVal

import scala.collection.JavaConverters._

class InferenceModel(private var autoScalingEnabled: Boolean = true,
                     private var concurrentNum: Int = 20,
                     private var originalModel: AbstractModel = null,
                     private[inference] var modelQueue:
                     LinkedBlockingQueue[AbstractModel] = null)
  extends InferenceSupportive with Serializable {

  require(concurrentNum > 0, "concurrentNum should > 0")

  /**
   * default constructor, will create a InferenceModel with auto-scaling enabled.
   *
   * @return an auto-scaling enabled InferenceModel
   */
  def this() = this(true, 20, null, null)

  /**
   * create an auto-scaling disabled InferenceModel with supportedConcurrentNum
   *
   * @param concurrentNum the concurrentNum of the InferenceModel
   * @return an auto-scaling disabled InferenceModel
   */
  def this(concurrentNum: Int) = this(false, concurrentNum, null, null)

  /**
   * create an InferenceModel with specified autoScalingEnabled, supportedConcurrentNum
   * and maxConcurrentNum
   *
   * @param autoScalingEnabled     if auto-scaling is enabled
   * @param concurrentNum          the concurrentNum of the InferenceModel
   * @return a specified InferenceModel
   */
  def this(autoScalingEnabled: Boolean, concurrentNum: Int) =
    this(autoScalingEnabled, concurrentNum, null, null)

  this.modelQueue = new LinkedBlockingQueue[AbstractModel](concurrentNum)

  this.originalModel match {
    case null =>
    case _ => offerModelQueue()
  }

  /**
   * loads a bigdl, analytics-zoo model
   *
   * @param modelPath  the file path of the model
   * @param weightPath the file path of the weights
   */
  def doLoad(modelPath: String, weightPath: String = null): Unit = {
    clearModelQueue()
    this.originalModel = InferenceModelFactory.loadFloatModel(modelPath, weightPath)
    offerModelQueue()
  }

  /**
   * loads a caffe model
   *
   * @param modelPath  the path of the prototxt file
   * @param weightPath the path of the caffemodel file
   */
  def doLoadCaffe(modelPath: String, weightPath: String): Unit = {
    clearModelQueue()
    this.originalModel = InferenceModelFactory.loadFloatModelForCaffe(modelPath, weightPath)
    offerModelQueue()
  }

  /**
   * loads a TF model as TFNet
   *
   * @param modelPath the path of the tensorflow model file
   */
  def doLoadTF(modelPath: String): Unit = {
    doLoadTensorflowModel(modelPath, 1, 1, true)
  }

  /**
   * loads a TF model as TFNet
   *
   * @param modelPath                 the path of the tensorflow model
   * @param intraOpParallelismThreads the num of intraOpParallelismThreads
   * @param interOpParallelismThreads the num of interOpParallelismThreads
   * @param usePerSessionThreads      whether to perSessionThreads
   */
  def doLoadTF(modelPath: String,
               intraOpParallelismThreads: Int,
               interOpParallelismThreads: Int,
               usePerSessionThreads: Boolean): Unit = {
    doLoadTensorflowModel(
      modelPath,
      intraOpParallelismThreads,
      interOpParallelismThreads,
      usePerSessionThreads)
  }

  /**
   * loads a TF model as OpenVINO
   *
   * @param modelPath the path of the tensorflow model
   * @param modelType the type of the tensorflow model,
   *                  please refer to [[ModelType]]
   *                  e.g. faster_rcnn_resnet101_coco, mask_rcnn_inception_v2_coco,
   *                  rfcn_resnet101_coco, ssd_inception_v2_coco
   */
  def doLoadTF(modelPath: String, modelType: String): Unit = {
    doLoadTensorflowModelAsOpenVINO(
      modelPath,
      modelType,
      null,
      null,
      DeviceType.CPU)
  }

  /**
   * loads a TF model as OpenVINO
   *
   * @param modelPath            the path of the tensorflow model
   * @param pipelineConfigPath   the path of the pipeline configure file
   * @param extensionsConfigPath the path of the extensions configure file
   */
  def doLoadTF(modelPath: String,
               pipelineConfigPath: String,
               extensionsConfigPath: String): Unit = {
    doLoadTensorflowModelAsOpenVINO(
      modelPath,
      null,
      pipelineConfigPath,
      extensionsConfigPath,
      DeviceType.CPU
    )
  }

  /**
   * loads a TF model as OpenVINO
   *
   * @param modelPath            the path of the tensorflow model
   * @param modelType            the type of the tensorflow model,
   *                             please refer to [[ModelType]]
   *                             e.g. faster_rcnn_resnet101_coco, mask_rcnn_inception_v2_coco,
   *                             rfcn_resnet101_coco, ssd_inception_v2_coco
   * @param pipelineConfigPath   the path of the pipeline configure file
   * @param extensionsConfigPath the path of the extensions configure file
   */
  def doLoadTF(modelPath: String,
               modelType: String,
               pipelineConfigPath: String,
               extensionsConfigPath: String): Unit = {
    doLoadTensorflowModelAsOpenVINO(
      modelPath,
      modelType,
      pipelineConfigPath,
      extensionsConfigPath,
      DeviceType.CPU
    )
  }

  /**
   * loads a openvino IR
   *
   * @param modelPath  the path of openvino ir xml file
   * @param weightPath the path of openvino ir bin file
   */
  def doLoadOpenVINO(modelPath: String, weightPath: String): Unit = {
    if (concurrentNum > 1) {
      InferenceSupportive.logger.warn(s"concurrentNum is $concurrentNum > 1, " +
        s"openvino model does not support shared weights model copies")
    }
    clearModelQueue()
    this.originalModel =
      InferenceModelFactory.loadOpenVINOModelForIR(modelPath, weightPath, DeviceType.CPU)
    offerModelQueue()
  }

  private def doLoadTensorflowModel(modelPath: String,
                                    intraOpParallelismThreads: Int,
                                    interOpParallelismThreads: Int,
                                    usePerSessionThreads: Boolean): Unit = {
    clearModelQueue()
    this.originalModel =
      InferenceModelFactory.loadFloatModelForTF(modelPath,
        intraOpParallelismThreads, interOpParallelismThreads, usePerSessionThreads)
    offerModelQueue()
  }

  private def doLoadTensorflowModelAsOpenVINO(modelPath: String,
                                              modelType: String,
                                              pipelineConfigPath: String,
                                              extensionsConfigPath: String,
                                              deviceType: DeviceTypeEnumVal): Unit = {
    if (concurrentNum > 1) {
      InferenceSupportive.logger.warn(s"concurrentNum is $concurrentNum > 1, " +
        s"openvino model does not support shared weights model copies")
    }
    clearModelQueue()
    this.originalModel = InferenceModelFactory.loadOpenVINOModelForTF(
      modelPath, modelType, pipelineConfigPath, extensionsConfigPath, deviceType)
    offerModelQueue()
  }

  /**
   * reloads the bigdl, analytics-zoo model
   *
   * @param modelPath  the file path of the model
   * @param weightPath the file path of the weights
   */
  def doReload(modelPath: String, weightPath: String): Unit = {
    clearModelQueue()
    doLoad(modelPath, weightPath)
  }

  @deprecated
  def doPredict(input: JList[JFloat], shape: JList[JInt]): JList[JFloat] = {
    timing("model predict") {
      val inputTensor = new JTensor(input, shape)
      val inputList = util.Arrays.asList({
        inputTensor
      })
      val inputs = util.Arrays.asList({
        inputList
      })
      val results = predict(inputs)
      results.get(0).get(0).getData.toList.asJava.asInstanceOf[JList[JFloat]]
    }
  }

  /**
   * predicts the inference result
   *
   * @param inputs the input tensor with batch
   * @return the output tensor with batch
   */
  def doPredict(inputs: JList[JList[JTensor]]): JList[JList[JTensor]] = {
    timing(s"model predict for batch ${inputs.size()}") {
      val batchSize = inputs.size()
      require(batchSize > 0, "inputs size should > 0")
      predict(inputs)
    }
  }

  /**
   * predicts the inference result
   *
   * @param inputActivity the input activity
   * @return the output activity
   */
  def doPredict(inputActivity: Activity): Activity = {
    timing(s"model predict for activity") {
      predict(inputActivity)
    }
  }

  /**
   * release original model and all the cloned ones in the queue
   */
  def release(): Unit = {
    clearModelQueue()
  }

  private def predict(inputActivity: Activity): Activity = {
    val model: AbstractModel = retrieveModel()
    try {
      model.predict(inputActivity)
    } finally {
      model match {
        case null =>
        case _ =>
          val success = modelQueue.offer(model)
          success match {
            case true =>
            case false => model.release()
          }
      }
    }
  }

  private def predict(inputs: JList[JList[JTensor]]): JList[JList[JTensor]] = {
    val model: AbstractModel = retrieveModel()
    try {
      model.predict(inputs)
    } finally {
      model match {
        case null =>
        case _ =>
          val success = modelQueue.offer(model)
          success match {
            case true =>
            case false => model.release()
          }
      }
    }
  }

  private def retrieveModel(): AbstractModel = {
    var model: AbstractModel = null
    autoScalingEnabled match {
      case false =>
        // if auto-scaling is not enabled, will take a model, waiting if necessary.
        try {
          model = modelQueue.take
        } catch {
          case e: InterruptedException =>
            throw new InferenceRuntimeException("no model available", e);
        }
      case true =>
        // if auto-scaling is enabled, will poll a model, no waiting but scale 1 model if necessary.
        model = modelQueue.poll()
        model match {
          case null => model = this.originalModel.copy(1)(0)
          case _ =>
        }
    }
    model
  }

  private def clearModelQueue(): Unit = {
    this.originalModel match {
      case null =>
      case _ => this.originalModel.release(); this.originalModel = null
    }
    List.range(0, this.modelQueue.size()).map(_ => {
      val model = this.modelQueue.take
      this.modelQueue.remove(model)
      model.release()
    })
    this.modelQueue.clear()
  }

  private def offerModelQueue(): Unit = {
    require(this.originalModel != null, "original model can not be null")
    require(this.concurrentNum > 0, "supported concurrent number should > 0")
    autoScalingEnabled match {
      case true =>
      case false =>
        val models = this.originalModel.copy(concurrentNum)
        models.map(this.modelQueue.offer(_))
    }
  }

  def getOriginalModel: AbstractModel = originalModel

  override def toString: String =
    s"InferenceModel($autoScalingEnabled, $concurrentNum, $originalModel, $modelQueue)"

}
