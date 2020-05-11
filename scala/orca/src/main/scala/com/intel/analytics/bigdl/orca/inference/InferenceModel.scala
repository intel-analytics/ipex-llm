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

import java.io.FileWriter
import java.lang.{Float => JFloat, Integer => JInt}
import java.util
import java.util.concurrent.LinkedBlockingQueue
import java.util.{List => JList}

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.sun.xml.internal.bind.v2.TODO

import scala.collection.JavaConverters._

class InferenceModel(private var autoScalingEnabled: Boolean = true,
                     private var concurrentNum: Int = 20,
                     private var originalModel: AbstractModel = null,
                     private[inference] var modelQueue:
                     LinkedBlockingQueue[AbstractModel] = null)
  extends InferenceSupportive with Serializable {

  require(concurrentNum > 0, "concurrentNum should > 0")

  private var batchCnt: Int = 0
  @transient var inferenceSummary: InferenceSummary = null
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
   * @param blas       whether MKLBLAS or MKLDNN
   */
  def doLoadBigDL(modelPath: String,
             weightPath: String = null,
             blas: Boolean = true): Unit = {
    clearModelQueue()
    this.originalModel = InferenceModelFactory.loadFloatModelForBigDL(modelPath, weightPath, blas)
    offerModelQueue()
  }

  /**
   * loads a bigdl, analytics-zoo model
   *
   * @param modelPath  the file path of the model
   * @param weightPath the file path of the weights
   * @param blas       whether MKLBLAS or MKLDNN
   */
  @deprecated("this method is deprecated, use doLoadBigDL() instead")
  def doLoad(modelPath: String,
             weightPath: String = null,
             blas: Boolean = true): Unit = {
    clearModelQueue()
    this.originalModel = InferenceModelFactory.loadFloatModelForBigDL(modelPath, weightPath, blas)
    offerModelQueue()
  }

  /**
   * loads a caffe model
   *
   * @param modelPath  the path of the prototxt file
   * @param weightPath the path of the caffemodel file
   * @param blas       whether MKLBLAS or MKLDNN
   */
  def doLoadCaffe(modelPath: String,
                  weightPath: String,
                  blas: Boolean = true): Unit = {
    clearModelQueue()
    this.originalModel = InferenceModelFactory.loadFloatModelForCaffe(modelPath, weightPath, blas)
    offerModelQueue()
  }

  /**
   * loads a tensorflow model as TFNet
   *
   * @param modelPath the path of the tensorflow frozen model
   * @param modelType the type of the tensorflow model file: "frozenModel"
   */
  def doLoadTensorflow(modelPath: String, modelType: String): Unit = {
    doLoadTensorflowModel(modelPath, modelType, 1, 1, true)
  }

  /**
   * loads a tensorflow model as TFNet
   *
   * @param modelPath                 the path of the tensorflow frozen model
   * @param modelType                 the type of the tensorflow model file: "frozenModel"
   * @param intraOpParallelismThreads the num of intraOpParallelismThreads
   * @param interOpParallelismThreads the num of interOpParallelismThreads
   * @param usePerSessionThreads      whether to perSessionThreads
   */
  def doLoadTensorflow(modelPath: String,
                       modelType: String,
                       intraOpParallelismThreads: Int,
                       interOpParallelismThreads: Int,
                       usePerSessionThreads: Boolean): Unit = {
    doLoadTensorflowModel(
      modelPath,
      modelType,
      intraOpParallelismThreads,
      interOpParallelismThreads,
      usePerSessionThreads)
  }

  /**
   * loads a tensorflow model as TFNet
   *
   * @param modelPath the path of the tensorflow frozen model file or saved model dir
   * @param modelType the type of the tensorflow model file: "frozenModel" or "savedModel"
   * @param inputs    the inputs of the model
   * @param outputs   the outputs of the model
   */
  def doLoadTensorflow(modelPath: String,
                       modelType: String,
                       inputs: Array[String],
                       outputs: Array[String]): Unit = {
    doLoadTensorflowModel(modelPath, modelType, inputs, outputs, 1, 1, true)
  }

  /**
   * loads a tensorflow model as TFNet
   *
   * @param modelPath                 the path of the tensorflow:
   *                                  frozen model file or saved model dir
   * @param modelType                 the type of the tensorflow model file:
   *                                  "frozenModel" or "savedModel"
   * @param inputs                    the inputs of the model
   * @param outputs                   the outputs of the model
   * @param intraOpParallelismThreads the num of intraOpParallelismThreads
   * @param interOpParallelismThreads the num of interOpParallelismThreads
   * @param usePerSessionThreads      whether to perSessionThreads
   */
  def doLoadTensorflow(modelPath: String,
                       modelType: String,
                       inputs: Array[String],
                       outputs: Array[String],
                       intraOpParallelismThreads: Int,
                       interOpParallelismThreads: Int,
                       usePerSessionThreads: Boolean): Unit = {
    doLoadTensorflowModel(
      modelPath,
      modelType,
      inputs,
      outputs,
      intraOpParallelismThreads,
      interOpParallelismThreads,
      usePerSessionThreads)
  }

  /**
   * loads a tensorflow model as TFNet
   *
   * @param modelBytes the bytes of the tensorflow model tar
   * @param modelType  the type of the tensorflow model file: "frozenModel" or "savedModel"
   * @param inputs     the inputs of the model
   * @param outputs    the outputs of the model
   */
  def doLoadTensorflow(modelBytes: Array[Byte],
                       modelType: String,
                       inputs: Array[String],
                       outputs: Array[String]): Unit = {
    doLoadTensorflowModel(modelBytes, modelType, inputs, outputs, 1, 1, true)
  }

  /**
   * loads a tensorflow model as TFNet
   *
   * @param modelBytes                the bytes of the tensorflow model tar
   * @param modelType                 the type of the tensorflow model file:
   *                                  "frozenModel" or "savedModel"
   * @param inputs                    the inputs of the model
   * @param outputs                   the outputs of the model
   * @param intraOpParallelismThreads the num of intraOpParallelismThreads
   * @param interOpParallelismThreads the num of interOpParallelismThreads
   * @param usePerSessionThreads      whether to perSessionThreads
   */
  def doLoadTensorflow(modelBytes: Array[Byte],
                       modelType: String,
                       inputs: Array[String],
                       outputs: Array[String],
                       intraOpParallelismThreads: Int,
                       interOpParallelismThreads: Int,
                       usePerSessionThreads: Boolean): Unit = {
    doLoadTensorflowModel(
      modelBytes,
      modelType,
      inputs,
      outputs,
      intraOpParallelismThreads,
      interOpParallelismThreads,
      usePerSessionThreads)
  }

  /**
   * load a Torch model as TorchNet
   *
   * @param modelPath the path of the torch script
   */
  def doLoadPyTorch(modelPath: String): Unit = {
    doLoadPyTorchModel(modelPath)
  }

  /**
   * load a Torch model as TorchNet
   *
   * @param modelBytes the bytes of the torch script
   */
  def doLoadPyTorch(modelBytes: Array[Byte]): Unit = {
    doLoadPyTorchModel(modelBytes)
  }

  /**
   * loads a TF model as OpenVINO
   *
   * @param modelPath the path of the tensorflow model
   * @param objectDetectionModelType the type of the tensorflow model,
   *                  please refer to [[ModelType]]
   *                  e.g. faster_rcnn_resnet101_coco, mask_rcnn_inception_v2_coco,
   *                  rfcn_resnet101_coco, ssd_inception_v2_coco
   */
  @deprecated("this method is deprecated", "0.8.0")
  def doLoadTF(modelPath: String, objectDetectionModelType: String): Unit = {
    doLoadTensorflowModelAsOpenVINO(
      modelPath,
      objectDetectionModelType,
      null,
      null)
  }

  /**
   * loads a TF model as OpenVINO
   *
   * @param modelPath            the path of the tensorflow model
   * @param pipelineConfigPath   the path of the pipeline configure file
   * @param extensionsConfigPath the path of the extensions configure file
   */
  @deprecated("this method is deprecated", "0.8.0")
  def doLoadTF(modelPath: String,
               pipelineConfigPath: String,
               extensionsConfigPath: String): Unit = {
    doLoadTensorflowModelAsOpenVINO(
      modelPath,
      null,
      pipelineConfigPath,
      extensionsConfigPath
    )
  }

  /**
   * loads a TF model as OpenVINO
   *
   * @param modelPath            the path of the tensorflow model
   * @param objectDetectionModelType  the type of the tensorflow model,
   *                             please refer to [[ModelType]]
   *                             e.g. faster_rcnn_resnet101_coco, mask_rcnn_inception_v2_coco,
   *                             rfcn_resnet101_coco, ssd_inception_v2_coco
   * @param pipelineConfigPath   the path of the pipeline configure file
   * @param extensionsConfigPath the path of the extensions configure file
   */
  @deprecated("this method is deprecated", "0.8.0")
  def doLoadTF(modelPath: String,
               objectDetectionModelType: String,
               pipelineConfigPath: String,
               extensionsConfigPath: String): Unit = {
    doLoadTensorflowModelAsOpenVINO(
      modelPath,
      objectDetectionModelType,
      pipelineConfigPath,
      extensionsConfigPath
    )
  }

  /**
   * load TF model as OpenVINO IR
   *
   * @param modelPath              the path of the tensorflow model
   * @param imageClassificationModelType the type of the tensorflow model
   * @param checkpointPath         the path of the tensorflow checkpoint file
   * @param inputShape             input shape that should be fed to an input node(s) of the model
   * @param ifReverseInputChannels the boolean value of if need reverse input channels.
   *                               switch the input channels order from RGB to BGR (or vice versa).
   * @param meanValues             all input values coming from original network inputs
   *                               will be divided by this value.
   * @param scale                  the scale value, to be used for the input image per channel.
   */
  @deprecated("this method is deprecated", "0.8.0")
  def doLoadTF(modelPath: String,
               imageClassificationModelType: String,
               checkpointPath: String,
               inputShape: Array[Int],
               ifReverseInputChannels: Boolean,
               meanValues: Array[Float],
               scale: Float): Unit = {
    doLoadTensorflowModelAsOpenVINO(
      modelPath, imageClassificationModelType, checkpointPath,
      inputShape, ifReverseInputChannels, meanValues, scale)
  }

  /**
   * load TF model as OpenVINO IR
   *
   * @param modelBytes             the bytes of the tensorflow model
   * @param imageClassificationModelType the type of the tensorflow model
   * @param checkpointBytes        the bytes of the tensorflow checkpoint file
   * @param inputShape             input shape that should be fed to an input node(s) of the model
   * @param ifReverseInputChannels the boolean value of if need reverse input channels.
   *                               switch the input channels order from RGB to BGR (or vice versa).
   * @param meanValues             all input values coming from original network inputs
   *                               will be divided by this value.
   * @param scale                  the scale value, to be used for the input image per channel.
   */
  @deprecated("this method is deprecated", "0.8.0")
  def doLoadTF(modelBytes: Array[Byte],
               imageClassificationModelType: String,
               checkpointBytes: Array[Byte],
               inputShape: Array[Int],
               ifReverseInputChannels: Boolean,
               meanValues: Array[Float],
               scale: Float): Unit = {
    doLoadTensorflowModelAsOpenVINO(
      modelBytes, imageClassificationModelType, checkpointBytes,
      inputShape, ifReverseInputChannels, meanValues, scale)
  }

  /**
   * load TF model as OpenVINO IR
   *
   * @param savedModelDir          the dir path of the tensorflow saved model
   * @param inputShape             input shape that should be fed to an input node(s) of the model
   * @param ifReverseInputChannels the boolean value of if need reverse input channels.
   *                               switch the input channels order from RGB to BGR (or vice versa).
   * @param meanValues             all input values coming from original network inputs
   *                               will be divided by this value.
   * @param scale                  the scale value, to be used for the input image per channel.
   * @param input                  the name of the input operation of the given model
   */
  @deprecated("this method is deprecated", "0.8.0")
  def doLoadTF(savedModelDir: String,
               inputShape: Array[Int],
               ifReverseInputChannels: Boolean,
               meanValues: Array[Float],
               scale: Float,
               input: String): Unit = {
    doLoadTensorflowModelAsOpenVINO(savedModelDir, inputShape,
      ifReverseInputChannels, meanValues, scale, input)
  }

  /**
   * load TF model as OpenVINO IR
   *
   * @param savedModelBytes        the bytes of the tensorflow saved model tar
   * @param inputShape             input shape that should be fed to an input node(s) of the model
   * @param ifReverseInputChannels the boolean value of if need reverse input channels.
   *                               switch the input channels order from RGB to BGR (or vice versa).
   * @param meanValues             all input values coming from original network inputs
   *                               will be divided by this value.
   * @param scale                  the scale value, to be used for the input image per channel.
   * @param input                  the name of the input operation of the given model
   */
  @deprecated("this method is deprecated", "0.8.0")
  def doLoadTF(savedModelBytes: Array[Byte],
               inputShape: Array[Int],
               ifReverseInputChannels: Boolean,
               meanValues: Array[Float],
               scale: Float,
               input: String): Unit = {
    doLoadTensorflowModelAsOpenVINO(savedModelBytes, inputShape,
      ifReverseInputChannels, meanValues, scale, input)
  }

  /**
   * loads a openvino IR
   *
   * @param modelPath  the path of openvino ir xml file
   * @param weightPath the path of openvino ir bin file
   */
  def doLoadOpenVINO(modelPath: String, weightPath: String, batchSize: Int = 0): Unit = {
    if (concurrentNum > 1) {
      InferenceSupportive.logger.warn(s"concurrentNum is $concurrentNum > 1, " +
        s"openvino model does not support shared weights model copies")
    }
    clearModelQueue()
    this.originalModel =
      InferenceModelFactory.loadOpenVINOModelForIR(modelPath, weightPath,
        DeviceType.CPU, batchSize)
    offerModelQueue()
  }

  /**
   * loads a openvino IR
   *
   * @param modelBytes  the bytes of openvino ir xml file
   * @param weightBytes the bytes of openvino ir bin file
   * @param batchSize   the batchsize of openvino ir
   */
  def doLoadOpenVINO(modelBytes: Array[Byte],
                     weightBytes: Array[Byte], batchSize: Int): Unit = {
    if (concurrentNum > 1) {
      InferenceSupportive.logger.warn(s"concurrentNum is $concurrentNum > 1, " +
        s"openvino model does not support shared weights model copies")
    }
    clearModelQueue()
    this.originalModel =
      InferenceModelFactory.loadOpenVINOModelForIR(modelBytes, weightBytes,
        DeviceType.CPU, batchSize)
    offerModelQueue()
  }

  private def doLoadTensorflowModel(modelPath: String,
                                    modelType: String,
                                    intraOpParallelismThreads: Int,
                                    interOpParallelismThreads: Int,
                                    usePerSessionThreads: Boolean): Unit = {
    modelType match {
      case null | "" =>
        require(modelType != null && modelType != "",
          "modelType should be specified as frozenModel")
      case "frozenModel" =>
        InferenceSupportive.logger.info(s"$modelType is supported.")
        doLoadTensorflowFrozenModel(modelPath,
          intraOpParallelismThreads, interOpParallelismThreads, usePerSessionThreads)
      case _ =>
        InferenceSupportive.logger.warn(s"$modelType not supported, " +
          s"supported tensorflow model file should be frozenModel")
    }
  }

  private def doLoadTensorflowModel(modelPath: String,
                                    modelType: String,
                                    inputs: Array[String],
                                    outputs: Array[String],
                                    intraOpParallelismThreads: Int,
                                    interOpParallelismThreads: Int,
                                    usePerSessionThreads: Boolean): Unit = {
    modelType match {
      case null | "" =>
        require(modelType != null && modelType != "",
          "modelType should be specified")
      case "frozenModel" =>
        InferenceSupportive.logger.info(s"$modelType is supported.")
        doLoadTensorflowFrozenModel(modelPath, inputs, outputs,
          intraOpParallelismThreads, interOpParallelismThreads, usePerSessionThreads)
      case "savedModel" =>
        InferenceSupportive.logger.info(s"$modelType is supported.")
        doLoadTensorflowSavedModel(modelPath, inputs, outputs,
          intraOpParallelismThreads, interOpParallelismThreads, usePerSessionThreads)
      case _ =>
        InferenceSupportive.logger.warn(s"$modelType not supported, " +
          s"supported tf model types are listed: " +
          s"${InferenceSupportive.modelType}")
    }
  }

  private def doLoadTensorflowModel(modelBytes: Array[Byte],
                                    modelType: String,
                                    inputs: Array[String],
                                    outputs: Array[String],
                                    intraOpParallelismThreads: Int,
                                    interOpParallelismThreads: Int,
                                    usePerSessionThreads: Boolean): Unit = {
    modelType match {
      case null | "" =>
        require(modelType != null && modelType != "",
          "modelType should be specified")
      case "frozenModel" =>
        InferenceSupportive.logger.info(s"$modelType is supported.")
        doLoadTensorflowFrozenModel(modelBytes, inputs, outputs,
          intraOpParallelismThreads, interOpParallelismThreads, usePerSessionThreads)
      case "savedModel" =>
        InferenceSupportive.logger.info(s"$modelType is supported.")
        doLoadTensorflowSavedModel(modelBytes, inputs, outputs,
          intraOpParallelismThreads, interOpParallelismThreads, usePerSessionThreads)
      case _ =>
        InferenceSupportive.logger.warn(s"$modelType not supported, " +
          s"supported tf model types are listed: " +
          s"${InferenceSupportive.modelType}")
    }
  }

  private def doLoadTensorflowFrozenModel(modelPath: String,
                                          intraOpParallelismThreads: Int,
                                          interOpParallelismThreads: Int,
                                          usePerSessionThreads: Boolean): Unit = {
    clearModelQueue()
    this.originalModel =
      InferenceModelFactory.loadFloatModelForTF(modelPath,
        intraOpParallelismThreads, interOpParallelismThreads, usePerSessionThreads)
    offerModelQueue()
  }

  private def doLoadTensorflowFrozenModel(modelPath: String,
                                          inputs: Array[String],
                                          outputs: Array[String],
                                          intraOpParallelismThreads: Int,
                                          interOpParallelismThreads: Int,
                                          usePerSessionThreads: Boolean): Unit = {
    clearModelQueue()
    this.originalModel =
      InferenceModelFactory.loadFloatModelForTFFrozenModel(modelPath,
        inputs, outputs, intraOpParallelismThreads, interOpParallelismThreads, usePerSessionThreads)
    offerModelQueue()
  }

  private def doLoadTensorflowFrozenModel(frozenModelBytes: Array[Byte],
                                          inputs: Array[String],
                                          outputs: Array[String],
                                          intraOpParallelismThreads: Int,
                                          interOpParallelismThreads: Int,
                                          usePerSessionThreads: Boolean): Unit = {
    clearModelQueue()
    this.originalModel =
      InferenceModelFactory.loadFloatModelForTFFrozenModelBytes(frozenModelBytes,
        inputs, outputs, intraOpParallelismThreads, interOpParallelismThreads, usePerSessionThreads)
    offerModelQueue()
  }

  private def doLoadTensorflowSavedModel(modelPath: String,
                                         inputs: Array[String],
                                         outputs: Array[String],
                                         intraOpParallelismThreads: Int,
                                         interOpParallelismThreads: Int,
                                         usePerSessionThreads: Boolean): Unit = {
    clearModelQueue()
    this.originalModel =
      InferenceModelFactory.loadFloatModelForTFSavedModel(modelPath,
        inputs, outputs, intraOpParallelismThreads, interOpParallelismThreads, usePerSessionThreads)
    offerModelQueue()
  }

  private def doLoadTensorflowSavedModel(savedModelBytes: Array[Byte],
                                         inputs: Array[String],
                                         outputs: Array[String],
                                         intraOpParallelismThreads: Int,
                                         interOpParallelismThreads: Int,
                                         usePerSessionThreads: Boolean): Unit = {
    clearModelQueue()
    this.originalModel =
      InferenceModelFactory.loadFloatModelForTFSavedModelBytes(savedModelBytes,
        inputs, outputs, intraOpParallelismThreads, interOpParallelismThreads, usePerSessionThreads)
    offerModelQueue()
  }

  private def doLoadPyTorchModel(modelPath: String): Unit = {
    clearModelQueue()
    this.originalModel = InferenceModelFactory.loadFloatModelForPyTorch(modelPath)
    offerModelQueue()
  }

  private def doLoadPyTorchModel(modelBytes: Array[Byte]): Unit = {
    clearModelQueue()
    this.originalModel = InferenceModelFactory.loadFloatModelForPyTorch(modelBytes)
    offerModelQueue()
  }

  private def doLoadTensorflowModelAsOpenVINO(modelPath: String,
                                              modelType: String,
                                              pipelineConfigPath: String,
                                              extensionsConfigPath: String): Unit = {
    if (concurrentNum > 1) {
      InferenceSupportive.logger.warn(s"concurrentNum is $concurrentNum > 1, " +
        s"openvino model does not support shared weights model copies")
    }
    clearModelQueue()
    this.originalModel = InferenceModelFactory.loadOpenVINOModelForTF(
      modelPath, modelType, pipelineConfigPath, extensionsConfigPath)
    offerModelQueue()
  }

  private def doLoadTensorflowModelAsOpenVINO(modelPath: String,
                                              imageClassificationModelType: String,
                                              checkpointPath: String,
                                              inputShape: Array[Int],
                                              ifReverseInputChannels: Boolean,
                                              meanValues: Array[Float],
                                              scale: Float): Unit = {
    if (concurrentNum > 1) {
      InferenceSupportive.logger.warn(s"concurrentNum is $concurrentNum > 1, " +
        s"openvino model does not support shared weights model copies")
    }
    clearModelQueue()
    this.originalModel = InferenceModelFactory.loadOpenVINOModelForTF(
      modelPath, imageClassificationModelType, checkpointPath,
      inputShape, ifReverseInputChannels, meanValues, scale)
    offerModelQueue()
  }

  private def doLoadTensorflowModelAsOpenVINO(modelBytes: Array[Byte],
                                              imageClassificationModelType: String,
                                              checkpointBytes: Array[Byte],
                                              inputShape: Array[Int],
                                              ifReverseInputChannels: Boolean,
                                              meanValues: Array[Float],
                                              scale: Float): Unit = {
    if (concurrentNum > 1) {
      InferenceSupportive.logger.warn(s"concurrentNum is $concurrentNum > 1, " +
        s"openvino model does not support shared weights model copies")
    }
    clearModelQueue()
    this.originalModel = InferenceModelFactory.loadOpenVINOModelForTF(
      modelBytes, imageClassificationModelType, checkpointBytes,
      inputShape, ifReverseInputChannels, meanValues, scale)
    offerModelQueue()
  }

  private def doLoadTensorflowModelAsOpenVINO(savedModelDir: String,
                                              inputShape: Array[Int],
                                              ifReverseInputChannels: Boolean,
                                              meanValues: Array[Float],
                                              scale: Float,
                                              input: String): Unit = {
    if (concurrentNum > 1) {
      InferenceSupportive.logger.warn(s"concurrentNum is $concurrentNum > 1, " +
        s"openvino model does not support shared weights model copies")
    }
    clearModelQueue()
    this.originalModel = InferenceModelFactory.loadOpenVINOModelForTF(
      savedModelDir, inputShape, ifReverseInputChannels, meanValues, scale, input)
    offerModelQueue()
  }

  private def doLoadTensorflowModelAsOpenVINO(savedModelBytes: Array[Byte],
                                              inputShape: Array[Int],
                                              ifReverseInputChannels: Boolean,
                                              meanValues: Array[Float],
                                              scale: Float,
                                              input: String): Unit = {
    if (concurrentNum > 1) {
      InferenceSupportive.logger.warn(s"concurrentNum is $concurrentNum > 1, " +
        s"openvino model does not support shared weights model copies")
    }
    clearModelQueue()
    this.originalModel = InferenceModelFactory.loadOpenVINOModelForTF(
      savedModelBytes, inputShape, ifReverseInputChannels, meanValues, scale, input)
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
    doLoadBigDL(modelPath, weightPath)
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
    val batchSize = inputs.size()
    require(batchSize > 0, "inputs size should > 0")
    timing(s"model predict batch size " + batchSize) {
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
    predict(inputActivity)
  }

  /**
   * release original model and all the cloned ones in the queue
   */
  def doRelease(): Unit = {
    clearModelQueue()
  }

  private def predict(inputActivity: Activity): Activity = {
    val model: AbstractModel = retrieveModel()
    try {
      val begin = System.nanoTime()
      val batchSize = if (inputActivity.isTensor) {
        inputActivity.toTensor[Float].size(1)
      } else {
        inputActivity.toTable(1).asInstanceOf[Tensor[Float]].size(1)
      }
      val result = model.predict(inputActivity)
      val end = System.nanoTime()

      val latency = end - begin
      val name = s"model predict for batch ${batchSize}"
      InferenceSupportive.logger.info(s"$name time elapsed [${latency/1e9} s, ${latency/1e6} ms].")

      result
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
    List.range(0, this.modelQueue.size()).foreach(_ => {
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

  def setInferenceSummary(value: InferenceSummary): this.type = {
    this.inferenceSummary = value
    this
  }


  def getOriginalModel: AbstractModel = originalModel

  override def toString: String =
    s"InferenceModel($autoScalingEnabled, $concurrentNum, $originalModel, $modelQueue)"


}

object InferenceModel {
  /**
   * optimize TF model as OpenVINO IR
   *
   * @param modelPath              the path of the tensorflow model
   * @param objectDetectionModelType  the type of the tensorflow model
   * @param pipelineConfigPath     the path of the pipeline configure file
   * @param extensionsConfigPath   the path of the extensions configure file
   * @param outputDir              the output dir
   */
  def doOptimizeTF(modelPath: String,
                   objectDetectionModelType: String,
                   pipelineConfigPath: String,
                   extensionsConfigPath: String,
                   outputDir: String): Unit = {
    OpenVinoInferenceSupportive.optimizeTFObjectDetectionModel(
      modelPath, objectDetectionModelType, pipelineConfigPath, extensionsConfigPath, outputDir)
  }

  /**
   * optimize TF model as OpenVINO IR
   *
   * @param modelPath              the path of the tensorflow model
   * @param imageClassificationModelType  the type of the tensorflow model
   * @param checkpointPath         the path of the tensorflow checkpoint file
   * @param inputShape             input shape that should be fed to an input node(s) of the model
   * @param ifReverseInputChannels the boolean value of if need reverse input channels.
   *                               switch the input channels order from RGB to BGR (or vice versa).
   * @param meanValues             all input values coming from original network inputs
   *                               will be divided by this value.
   * @param scale                  the scale value, to be used for the input image per channel.
   * @param outputDir              the output dir
   */
  def doOptimizeTF(modelPath: String,
                   imageClassificationModelType: String,
                   checkpointPath: String,
                   inputShape: Array[Int],
                   ifReverseInputChannels: Boolean,
                   meanValues: Array[Float],
                   scale: Float,
                   outputDir: String): Unit = {
    OpenVinoInferenceSupportive.optimizeTFImageClassificationModel(
      modelPath, imageClassificationModelType, checkpointPath, inputShape,
      ifReverseInputChannels, meanValues, scale, outputDir)
  }
}
