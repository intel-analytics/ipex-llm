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

package com.intel.analytics.bigdl.python.api

import java.util.{List => JList}

import com.intel.analytics.bigdl.{Criterion, DataSet, nn}
import com.intel.analytics.bigdl.dataset.{DataSet, LocalDataSet, MiniBatch}
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.{Container, SpatialBatchNormalization}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.keras._
import com.intel.analytics.bigdl.numeric._
import com.intel.analytics.bigdl.optim.{OptimMethod, Regularizer, ValidationMethod}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.{ImageFeature, ImageFeatureToMiniBatch}
import com.intel.analytics.bigdl.utils.{Engine, MultiShape, Shape, SingleShape}
import org.apache.spark.api.java.JavaRDD

import scala.collection.JavaConverters._
import scala.language.existentials
import scala.reflect.ClassTag


object PythonBigDLKeras {

  def ofFloat(): PythonBigDLKeras[Float] = new PythonBigDLKeras[Float]()

  def ofDouble(): PythonBigDLKeras[Double] = new PythonBigDLKeras[Double]()
}

class PythonBigDLKeras[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonBigDL[T] {

  def toScalaShape(inputShape: JList[Int]): Shape = {
    if (inputShape == null) {
      null
    } else {
      Shape(inputShape.asScala.toArray)
    }
  }

  def toScalaMultiShape(inputShape: JList[JList[Int]]): Shape = {
    if (inputShape == null) {
      null
    } else {
      Shape(inputShape.asScala.toArray.map(shape => Shape(shape.asScala.toArray)).toList)
    }
  }

  def toScalaArray(list: JList[Int]): Array[Int] = {
    if (list == null) {
      null
    } else {
      list.asScala.toArray
    }
  }

  def createKerasModel(input: JList[ModuleNode[T]],
      output: JList[ModuleNode[T]]): Model[T] = {
      nn.keras.Model(input.asScala.toArray, output.asScala.toArray)
  }

  def createKerasSequential(): nn.keras.Sequential[T] = {
      nn.keras.Sequential[T]()
  }

  def createKerasInput(
    name : String = null,
    inputShape: JList[Int] = null): ModuleNode[T] = {
    Input(name = name, inputShape = toScalaShape(inputShape))
  }

  def createKerasInputLayer(
    inputShape: JList[Int] = null): KerasLayer[Activity, Activity, T] = {
    InputLayer(inputShape = toScalaShape(inputShape))
  }

  def shapeToJList(shape: Shape): JList[JList[Int]] = {
    val shapes = if (shape.isInstanceOf[SingleShape]) {
      MultiShape(List(shape))
    }
    else {
      shape
    }
    shapes.toMulti().map(single => single.toSingle().toList.asJava).toList.asJava
  }

  def getOutputShape(module: Container[Activity, Activity, T]): JList[JList[Int]] = {
    val output = module.getOutputShape()
    shapeToJList(output)
  }

  def getInputShape(module: Container[Activity, Activity, T]): JList[JList[Int]] = {
    val input = module.getInputShape()
    // TODO: inputShape can be nested MultiShape
    shapeToJList(input)
  }

  def createKerasDense(
    outputDim: Int,
    init: String = "glorot_uniform",
    activation: String = null,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    bias: Boolean = true,
    inputShape: JList[Int] = null): Dense[T] = {
    Dense(outputDim, init, activation, wRegularizer,
      bRegularizer, bias, toScalaShape(inputShape))
  }

  def createKerasEmbedding(
    inputDim: Int,
    outputDim: Int,
    init: String = "uniform",
    wRegularizer: Regularizer[T] = null,
    inputShape: JList[Int] = null): Embedding[T] = {
    Embedding[T](inputDim, outputDim, init, wRegularizer, toScalaShape(inputShape))
  }

  def createKerasBatchNormalization(
    epsilon: Double = 0.001,
    momentum: Double = 0.99,
    betaInit: String = "zero",
    gammaInit: String = "one",
    dimOrdering: String = "th",
    inputShape: JList[Int] = null): BatchNormalization[T] = {
    BatchNormalization[T](epsilon, momentum, betaInit,
      gammaInit, dimOrdering, toScalaShape(inputShape))
  }

  def setRunningMean(module: BatchNormalization[T], runningMean: JTensor): Unit = {
    module.labor.asInstanceOf[SpatialBatchNormalization[T]]
      .runningMean.set(toTensor(runningMean))
  }

  def setRunningStd(module: BatchNormalization[T], runningStd: JTensor): Unit = {
    module.labor.asInstanceOf[SpatialBatchNormalization[T]]
      .runningVar.set(toTensor(runningStd))
  }

  def getRunningMean(module: BatchNormalization[T]): JTensor = {
    toJTensor(module.labor.asInstanceOf[SpatialBatchNormalization[T]]
      .runningMean)
  }

  def getRunningStd(module: BatchNormalization[T]): JTensor = {
    toJTensor(module.labor.asInstanceOf[SpatialBatchNormalization[T]]
      .runningVar)
  }

  def createKerasMerge(
    layers: JList[AbstractModule[Activity, Activity, T]] = null,
    mode: String = "sum",
    concatAxis: Int = -1,
    inputShape: JList[JList[Int]]): Merge[T] = {
    val layersList = if (layers != null) layers.asScala.toList
                     else null
    Merge[T](layersList, mode, concatAxis, toScalaMultiShape(inputShape))
  }

  def createKerasConvolution2D(
    nbFilter: Int,
    nbRow: Int,
    nbCol: Int,
    init: String = "glorot_uniform",
    activation: String = null,
    borderMode: String = "valid",
    subsample: JList[Int],
    dimOrdering: String = "th",
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    bias: Boolean = true,
    inputShape: JList[Int] = null): Convolution2D[T] = {
    new Convolution2D(nbFilter, nbRow, nbCol, KerasUtils.getInitMethod(init),
      KerasUtils.getKerasActivation(activation), borderMode,
      toScalaArray(subsample), KerasUtils.toBigDLFormat(dimOrdering),
      wRegularizer, bRegularizer, bias, toScalaShape(inputShape))
  }

  def createKerasMaxPooling2D(
    poolSize: JList[Int],
    strides: JList[Int],
    borderMode: String = "valid",
    dimOrdering: String = "th",
    inputShape: JList[Int] = null): MaxPooling2D[T] = {
    new MaxPooling2D[T](toScalaArray(poolSize), toScalaArray(strides),
      borderMode, KerasUtils.toBigDLFormat(dimOrdering), toScalaShape(inputShape))
  }

  def createKerasActivation(
    activation: String,
    inputShape: JList[Int] = null): Activation[T] = {
    Activation(activation, toScalaShape(inputShape))
  }

  def createKerasReshape(
    targetShape: JList[Int],
    inputShape: JList[Int] = null): Reshape[T] = {
    Reshape(toScalaArray(targetShape), toScalaShape(inputShape))
  }

  def createKerasDropout(
    p: Double,
    inputShape: JList[Int] = null): Dropout[T] = {
    Dropout(p, toScalaShape(inputShape))
  }

  def createKerasFlatten(
    inputShape: JList[Int] = null): Flatten[T] = {
    Flatten(toScalaShape(inputShape))
  }

  def createKerasSimpleRNN(
    outputDim: Int,
    activation: String = "tanh",
    returnSequences: Boolean = false,
    goBackwards: Boolean = false,
    wRegularizer: Regularizer[T] = null,
    uRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    inputShape: JList[Int] = null): SimpleRNN[T] = {
    SimpleRNN(outputDim, activation, returnSequences, goBackwards,
      wRegularizer, uRegularizer, bRegularizer, toScalaShape(inputShape))
  }

  def createKerasLSTM(
    outputDim: Int,
    activation: String = "tanh",
    innerActivation: String = "hard_sigmoid",
    returnSequences: Boolean = false,
    goBackwards: Boolean = false,
    wRegularizer: Regularizer[T] = null,
    uRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    inputShape: JList[Int] = null): LSTM[T] = {
    LSTM(outputDim, activation, innerActivation, returnSequences,
      goBackwards, wRegularizer, uRegularizer, bRegularizer, toScalaShape(inputShape))
  }

  def createKerasGRU(
    outputDim: Int,
    activation: String = "tanh",
    innerActivation: String = "hard_sigmoid",
    returnSequences: Boolean = false,
    goBackwards: Boolean = false,
    wRegularizer: Regularizer[T] = null,
    uRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    inputShape: JList[Int] = null): GRU[T] = {
    GRU(outputDim, activation, innerActivation, returnSequences,
      goBackwards, wRegularizer, uRegularizer, bRegularizer, toScalaShape(inputShape))
  }

  def createKerasHighway(
    activation: String = null,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    bias: Boolean = true,
    inputShape: JList[Int] = null): Highway[T] = {
    Highway(activation, wRegularizer, bRegularizer, bias, toScalaShape(inputShape))
  }

  def createKerasZeroPadding1D(
    padding: JList[Int],
    inputShape: JList[Int] = null): ZeroPadding1D[T] = {
    new ZeroPadding1D(toScalaArray(padding), toScalaShape(inputShape))
  }

  def createKerasZeroPadding2D(
    padding: JList[Int],
    dimOrdering: String = "th",
    inputShape: JList[Int] = null): ZeroPadding2D[T] = {
    new ZeroPadding2D(toScalaArray(padding),
      KerasUtils.toBigDLFormat(dimOrdering), toScalaShape(inputShape))
  }

  def createKerasUpSampling1D(
    length: Int = 2,
    inputShape: JList[Int] = null): UpSampling1D[T] = {
    UpSampling1D(length, toScalaShape(inputShape))
  }

  def createKerasUpSampling2D(
    size: JList[Int],
    dimOrdering: String = "th",
    inputShape: JList[Int] = null): UpSampling2D[T] = {
    new UpSampling2D(toScalaArray(size), KerasUtils.toBigDLFormat(dimOrdering),
      toScalaShape(inputShape))
  }

  def createKerasUpSampling3D(
    size: JList[Int],
    dimOrdering: String = "th",
    inputShape: JList[Int] = null): UpSampling3D[T] = {
    new UpSampling3D(toScalaArray(size), KerasUtils.toBigDLFormat5D(dimOrdering),
      toScalaShape(inputShape))
  }

  def createKerasMaxoutDense(
    outputDim: Int,
    nbFeature: Int = 4,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    bias: Boolean = true,
    inputShape: JList[Int] = null): MaxoutDense[T] = {
    MaxoutDense(outputDim, nbFeature, wRegularizer,
      bRegularizer, bias, toScalaShape(inputShape))
  }

  def createKerasConvolution1D(
    nbFilter: Int,
    filterLength: Int,
    init: String = "glorot_uniform",
    activation: String = null,
    borderMode: String = "valid",
    subsampleLength: Int = 1,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    bias: Boolean = true,
    inputShape: JList[Int] = null): Convolution1D[T] = {
    Convolution1D(nbFilter, filterLength, init, activation, borderMode,
      subsampleLength, wRegularizer, bRegularizer, bias, toScalaShape(inputShape))
  }

  def createKerasConvolution3D(
    nbFilter: Int,
    kernelDim1: Int,
    kernelDim2: Int,
    kernelDim3: Int,
    init: String = "glorot_uniform",
    activation: String = null,
    borderMode: String = "valid",
    subsample: JList[Int],
    dimOrdering: String = "th",
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    bias: Boolean = true,
    inputShape: JList[Int] = null): Convolution3D[T] = {
    new Convolution3D(nbFilter, kernelDim1, kernelDim2, kernelDim3,
      KerasUtils.getInitMethod(init), KerasUtils.getKerasActivation(activation),
      borderMode, toScalaArray(subsample), KerasUtils.toBigDLFormat5D(dimOrdering),
      wRegularizer, bRegularizer, bias, toScalaShape(inputShape))
  }

  def createKerasMaxPooling1D(
    poolLength: Int = 2,
    stride: Int = -1,
    borderMode: String = "valid",
    inputShape: JList[Int] = null): MaxPooling1D[T] = {
    MaxPooling1D(poolLength, stride, borderMode, toScalaShape(inputShape))
  }

  def createKerasMaxPooling3D(
    poolSize: JList[Int],
    strides: JList[Int],
    dimOrdering: String = "th",
    inputShape: JList[Int] = null): MaxPooling3D[T] = {
    new MaxPooling3D(toScalaArray(poolSize), toScalaArray(strides),
      KerasUtils.toBigDLFormat5D(dimOrdering), toScalaShape(inputShape))
  }

  def createKerasAveragePooling1D(
    poolLength: Int = 2,
    stride: Int = -1,
    borderMode: String = "valid",
    inputShape: JList[Int] = null): AveragePooling1D[T] = {
    AveragePooling1D(poolLength, stride, borderMode, toScalaShape(inputShape))
  }

  def createKerasAveragePooling2D(
    poolSize: JList[Int],
    strides: JList[Int],
    borderMode: String = "valid",
    dimOrdering: String = "th",
    inputShape: JList[Int] = null): AveragePooling2D[T] = {
    new AveragePooling2D(toScalaArray(poolSize), toScalaArray(strides),
      borderMode, KerasUtils.toBigDLFormat(dimOrdering), toScalaShape(inputShape))
  }

  def createKerasAveragePooling3D(
    poolSize: JList[Int],
    strides: JList[Int],
    dimOrdering: String = "th",
    inputShape: JList[Int] = null): AveragePooling3D[T] = {
    new AveragePooling3D(toScalaArray(poolSize), toScalaArray(strides),
      KerasUtils.toBigDLFormat5D(dimOrdering), toScalaShape(inputShape))
  }

  def createKerasGlobalAveragePooling2D(
    dimOrdering: String = "th",
    inputShape: JList[Int] = null): GlobalAveragePooling2D[T] = {
    GlobalAveragePooling2D(dimOrdering, toScalaShape(inputShape))
  }

  def createKerasGlobalMaxPooling2D(
    dimOrdering: String = "th",
    inputShape: JList[Int] = null): GlobalMaxPooling2D[T] = {
    GlobalMaxPooling2D(dimOrdering, toScalaShape(inputShape))
  }

  def createKerasRepeatVector(
    n: Int,
    inputShape: JList[Int] = null): RepeatVector[T] = {
    RepeatVector(n, toScalaShape(inputShape))
  }

  def createKerasPermute(
    dims: JList[Int],
    inputShape: JList[Int] = null): Permute[T] = {
    Permute(toScalaArray(dims), toScalaShape(inputShape))
  }

  def createKerasCropping1D(
    cropping: JList[Int],
    inputShape: JList[Int] = null): Cropping1D[T] = {
    new Cropping1D(toScalaArray(cropping), toScalaShape(inputShape))
  }

  def createKerasCropping2D(
    heightCrop: JList[Int],
    widthCrop: JList[Int],
    dimOrdering: String = "th",
    inputShape: JList[Int] = null): Cropping2D[T] = {
    new Cropping2D(toScalaArray(heightCrop), toScalaArray(widthCrop),
      KerasUtils.toBigDLFormat(dimOrdering), toScalaShape(inputShape))
  }

  def createKerasCropping3D(
    dim1Crop: JList[Int],
    dim2Crop: JList[Int],
    dim3Crop: JList[Int],
    dimOrdering: String = "th",
    inputShape: JList[Int] = null): Cropping3D[T] = {
    new Cropping3D(toScalaArray(dim1Crop), toScalaArray(dim2Crop), toScalaArray(dim3Crop),
      KerasUtils.toBigDLFormat5D(dimOrdering), toScalaShape(inputShape))
  }

  def createKerasAtrousConvolution1D(
    nbFilter: Int,
    filterLength: Int,
    init: String = "glorot_uniform",
    activation: String = null,
    subsampleLength: Int = 1,
    atrousRate: Int = 1,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    inputShape: JList[Int] = null): AtrousConvolution1D[T] = {
    AtrousConvolution1D(nbFilter, filterLength, init, activation,
      subsampleLength, atrousRate, wRegularizer, bRegularizer, toScalaShape(inputShape))
  }

  def createKerasAtrousConvolution2D(
    nbFilter: Int,
    nbRow: Int,
    nbCol: Int,
    init: String = "glorot_uniform",
    activation: String = null,
    subsample: JList[Int],
    atrousRate: JList[Int],
    dimOrdering: String = "th",
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    inputShape: JList[Int] = null): AtrousConvolution2D[T] = {
    new AtrousConvolution2D(nbFilter, nbRow, nbCol, KerasUtils.getInitMethod(init),
      KerasUtils.getKerasActivation(activation), toScalaArray(subsample),
      toScalaArray(atrousRate), KerasUtils.toBigDLFormat(dimOrdering),
      wRegularizer, bRegularizer, toScalaShape(inputShape))
  }

  def createKerasDeconvolution2D(
    nbFilter: Int,
    nbRow: Int,
    nbCol: Int,
    init: String = "glorot_uniform",
    activation: String = null,
    subsample: JList[Int],
    dimOrdering: String = "th",
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    bias: Boolean = true,
    inputShape: JList[Int] = null): Deconvolution2D[T] = {
    new Deconvolution2D(nbFilter, nbRow, nbCol, KerasUtils.getInitMethod(init),
      KerasUtils.getKerasActivation(activation), toScalaArray(subsample),
      KerasUtils.toBigDLFormat(dimOrdering), wRegularizer, bRegularizer,
      bias, toScalaShape(inputShape))
  }

  def createKerasConvLSTM2D(
    nbFilter: Int,
    nbKernel: Int,
    activation: String = "tanh",
    innerActivation: String = "hard_sigmoid",
    dimOrdering: String = "th",
    subsample: Int = 1,
    wRegularizer: Regularizer[T] = null,
    uRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    returnSequences: Boolean = false,
    goBackwards: Boolean = false,
    inputShape: JList[Int] = null): ConvLSTM2D[T] = {
    ConvLSTM2D(nbFilter, nbKernel, activation, innerActivation,
      dimOrdering, subsample, wRegularizer, uRegularizer, bRegularizer,
      returnSequences, goBackwards, toScalaShape(inputShape))
  }

  def createKerasLocallyConnected1D(
    nbFilter: Int,
    filterLength: Int,
    activation: String = null,
    subsampleLength: Int = 1,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    bias: Boolean = true,
    inputShape: JList[Int] = null): LocallyConnected1D[T] = {
    LocallyConnected1D(nbFilter, filterLength, activation, subsampleLength,
      wRegularizer, bRegularizer, bias, toScalaShape(inputShape))
  }

  def createKerasLocallyConnected2D(
    nbFilter: Int,
    nbRow: Int,
    nbCol: Int,
    activation: String = null,
    borderMode: String = "valid",
    subsample: JList[Int],
    dimOrdering: String = "th",
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    bias: Boolean = true,
    inputShape: JList[Int] = null): LocallyConnected2D[T] = {
    new LocallyConnected2D(nbFilter, nbRow, nbCol, KerasUtils.getKerasActivation(activation),
      borderMode, toScalaArray(subsample), KerasUtils.toBigDLFormat(dimOrdering),
      wRegularizer, bRegularizer, bias, toScalaShape(inputShape))
  }

  def createKerasSeparableConvolution2D(
    nbFilter: Int,
    nbRow: Int,
    nbCol: Int,
    init: String = "glorot_uniform",
    activation: String = null,
    borderMode: String = "valid",
    subsample: JList[Int],
    depthMultiplier: Int = 1,
    dimOrdering: String = "th",
    depthwiseRegularizer: Regularizer[T] = null,
    pointwiseRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    bias: Boolean = true,
    inputShape: JList[Int] = null): SeparableConvolution2D[T] = {
    new SeparableConvolution2D(nbFilter, nbRow, nbCol, KerasUtils.getInitMethod(init),
      KerasUtils.getKerasActivation(activation), borderMode, toScalaArray(subsample),
      depthMultiplier, KerasUtils.toBigDLFormat(dimOrdering),
      depthwiseRegularizer, pointwiseRegularizer, bRegularizer, bias, toScalaShape(inputShape))
  }

  def createKerasZeroPadding3D(
    padding: JList[Int],
    dimOrdering: String = "th",
    inputShape: JList[Int] = null): ZeroPadding3D[T] = {
    new ZeroPadding3D(toScalaArray(padding), KerasUtils.toBigDLFormat5D(dimOrdering),
      toScalaShape(inputShape))
  }

  def createKerasGlobalAveragePooling1D(
    inputShape: JList[Int] = null): GlobalAveragePooling1D[T] = {
    GlobalAveragePooling1D(toScalaShape(inputShape))
  }

  def createKerasGlobalMaxPooling1D(
    inputShape: JList[Int] = null): GlobalMaxPooling1D[T] = {
    GlobalMaxPooling1D(toScalaShape(inputShape))
  }

  def createKerasGlobalMaxPooling3D(
    dimOrdering: String = "th",
    inputShape: JList[Int] = null): GlobalMaxPooling3D[T] = {
    GlobalMaxPooling3D(dimOrdering, toScalaShape(inputShape))
  }

  def createKerasGlobalAveragePooling3D(
    dimOrdering: String = "th",
    inputShape: JList[Int] = null): GlobalAveragePooling3D[T] = {
    GlobalAveragePooling3D(dimOrdering, toScalaShape(inputShape))
  }

  def createKerasSpatialDropout1D(
    p: Double = 0.5,
    inputShape: JList[Int] = null): SpatialDropout1D[T] = {
    SpatialDropout1D(p, toScalaShape(inputShape))
  }

  def createKerasSpatialDropout2D(
    p: Double = 0.5,
    dimOrdering: String = "th",
    inputShape: JList[Int] = null): SpatialDropout2D[T] = {
    SpatialDropout2D(p, dimOrdering, toScalaShape(inputShape))
  }

  def createKerasSpatialDropout3D(
    p: Double = 0.5,
    dimOrdering: String = "th",
    inputShape: JList[Int] = null): SpatialDropout3D[T] = {
    SpatialDropout3D(p, dimOrdering, toScalaShape(inputShape))
  }

  def createKerasGaussianDropout(
    p: Double,
    inputShape: JList[Int] = null): GaussianDropout[T] = {
    GaussianDropout(p, toScalaShape(inputShape))
  }

  def createKerasGaussianNoise(
    sigma: Double,
    inputShape: JList[Int] = null): GaussianNoise[T] = {
    GaussianNoise(sigma, toScalaShape(inputShape))
  }

  def createKerasMasking(
    maskValue: Double = 0.0,
    inputShape: JList[Int] = null): Masking[T] = {
    Masking(maskValue, toScalaShape(inputShape))
  }

  def createKerasSReLU(
    tLeftInit: String = "zero",
    aLeftInit: String = "glorot_uniform",
    tRightInit: String = "glorot_uniform",
    aRightInit: String = "one",
    sharedAxes: JList[Int] = null,
    inputShape: JList[Int] = null): SReLU[T] = {
    SReLU(tLeftInit, aLeftInit, tRightInit, aRightInit,
      toScalaArray(sharedAxes), toScalaShape(inputShape))
  }

  def createKerasELU(
    alpha: Double = 1.0,
    inputShape: JList[Int] = null): ELU[T] = {
    ELU(alpha, toScalaShape(inputShape))
  }

  def createKerasLeakyReLU(
    alpha: Double = 0.01,
    inputShape: JList[Int] = null): LeakyReLU[T] = {
    LeakyReLU(alpha, toScalaShape(inputShape))
  }

  def createKerasThresholdedReLU(
    theta: Double = 1.0,
    inputShape: JList[Int] = null): ThresholdedReLU[T] = {
    ThresholdedReLU(theta, toScalaShape(inputShape))
  }

  def createKerasTimeDistributed(
    layer: KerasLayer[Tensor[T], Tensor[T], T],
    inputShape: JList[Int] = null): TimeDistributed[T] = {
    TimeDistributed(layer, toScalaShape(inputShape))
  }

  def createKerasBidirectional(
    layer: Recurrent[T],
    mergeMode: String = "concat",
    inputShape: JList[Int] = null): Bidirectional[T] = {
    Bidirectional(layer, mergeMode, toScalaShape(inputShape))
  }

  def compile(
    module: KerasModel[T],
    optimizer: OptimMethod[T],
    loss: Criterion[T],
    metrics: JList[ValidationMethod[T]] = null): Unit = {
    module.compile(optimizer, loss,
      if (metrics == null) null else metrics.asScala.toArray)
  }

  def fit(
    module: KerasModel[T],
    x: JavaRDD[Sample],
    batchSize: Int = 32,
    epochs: Int = 10,
    validationData: JavaRDD[Sample] = null): Unit = {
    module.fit(toJSample(x), batchSize, epochs,
      if (validationData == null) null else toJSample(validationData))
  }

  def fit(
    module: KerasModel[T],
    x: DataSet[ImageFeature],
    batchSize: Int,
    epochs: Int,
    validationData: DataSet[ImageFeature]): Unit = {
    val trainData = x -> ImageFeatureToMiniBatch[T](batchSize)
    val valData =
      if (validationData != null) validationData -> ImageFeatureToMiniBatch[T](batchSize)
      else null
    module.fit(trainData, epochs, valData)
  }

  def fit(
    module: KerasModel[T],
    xTrain: JList[JTensor],
    yTrain: JTensor,
    batchSize: Int,
    epochs: Int,
    xVal: JList[JTensor],
    yVal: JTensor,
    localCores: Int): Unit = {
    val trainArray = toSampleArray(xTrain.asScala.toList.map{f => toTensor(f)}, toTensor(yTrain))
    val trainData = batching(DataSet.array(trainArray), batchSize)
      .asInstanceOf[LocalDataSet[MiniBatch[T]]]
    val valData = if (xVal != null && yVal != null) {
      val valArray = toSampleArray(xVal.asScala.toList.map{f => toTensor(f)}, toTensor(yVal))
      batching(DataSet.array(valArray), batchSize)
    } else null
    Engine.setNodeAndCore(1, localCores)
    module.fit(trainData, epochs, valData)
  }

  def evaluate(
    module: KerasModel[T],
    x: JavaRDD[Sample],
    batchSize: Int = 32): JList[EvaluatedResult] = {
    val resultArray = module.evaluate(toJSample(x), batchSize)
    val testResultArray = resultArray.map { result =>
      EvaluatedResult(result._1.result()._1, result._1.result()._2,
        result._2.toString())
    }
    testResultArray.toList.asJava
  }

}
