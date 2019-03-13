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

class InferenceModel(private var supportedConcurrentNum: Int = 1,
                     private var originalModel: AbstractModel = null,
                     private[inference] var modelQueue:
                     LinkedBlockingQueue[AbstractModel] = null)
  extends InferenceSupportive with Serializable {
  this.modelQueue = new LinkedBlockingQueue[AbstractModel](supportedConcurrentNum)
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
    if (supportedConcurrentNum > 1) {
      InferenceSupportive.logger.warn(s"supportedConcurrentNum is $supportedConcurrentNum > 1, " +
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
    if (supportedConcurrentNum > 1) {
      InferenceSupportive.logger.warn(s"supportedConcurrentNum is $supportedConcurrentNum > 1, " +
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
    var model: AbstractModel = null
    try {
      model = modelQueue.take
    } catch {
      case e: InterruptedException => throw new InferenceRuntimeException("no model available", e);
    }
    try {
      val result = model.predict(inputActivity)
      result
    } finally {
      modelQueue.offer(model)
    }
  }

  private def predict(inputs: JList[JList[JTensor]]): JList[JList[JTensor]] = {
    var model: AbstractModel = null
    try {
      model = modelQueue.take
    } catch {
      case e: InterruptedException => throw new InferenceRuntimeException("no model available", e);
    }
    try {
      model.predict(inputs)
    } finally {
      modelQueue.offer(model)
    }
  }

  private def clearModelQueue(): Unit = {
    this.originalModel match {
      case null =>
      case _ => this.originalModel.release(); this.originalModel = null
    }
    List.range(0, this.modelQueue.size()).map(i => {
      val model = this.modelQueue.take
      this.modelQueue.remove(model)
      model.release()
    })
    this.modelQueue.clear()
  }

  private def offerModelQueue(): Unit = {
    require(this.originalModel != null, "original model can not be null")
    require(this.supportedConcurrentNum > 0, "supported concurrent number should > 0")
    val models = this.originalModel.copy(supportedConcurrentNum)
    models.map(this.modelQueue.offer(_))
  }

  def getOriginalModel: AbstractModel = originalModel

  override def toString: String =
    s"InferenceModel($supportedConcurrentNum, $originalModel, $modelQueue)"

}
