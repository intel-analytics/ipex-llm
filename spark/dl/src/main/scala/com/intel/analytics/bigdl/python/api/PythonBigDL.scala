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

import java.util.{ArrayList => JArrayList, HashMap => JHashMap, List => JList, Map => JMap}

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{Identity => DIdentity, Sample => JSample, _}
import com.intel.analytics.bigdl.nn.{PGCriterion, Sequential, Zeros, _}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, _}
import com.intel.analytics.bigdl.numeric._
import com.intel.analytics.bigdl.optim.{Optimizer, _}
import com.intel.analytics.bigdl.tensor.{DenseType, SparseType, Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Table, _}
import com.intel.analytics.bigdl.visualization.{Summary, TrainSummary, ValidationSummary}
import org.apache.spark.api.java.{JavaRDD, JavaSparkContext}
import org.apache.spark.rdd.RDD
import java.lang.{Boolean => JBoolean}
import java.nio.ByteOrder

import com.intel.analytics.bigdl.dataset.image.{CropCenter, CropRandom, CropperMethod}
import com.intel.analytics.bigdl.dlframes._
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn.keras.{KerasLayer, KerasModel}
import com.intel.analytics.bigdl.optim.SGD.{LearningRateSchedule, SequentialSchedule}
import com.intel.analytics.bigdl.transform.vision.image._
import com.intel.analytics.bigdl.transform.vision.image.augmentation._
import com.intel.analytics.bigdl.transform.vision.image.label.roi._
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.bigdl.utils.tf.TensorflowDataFormat
import com.intel.analytics.bigdl.utils.tf.TensorflowLoader.parse
import com.intel.analytics.bigdl.utils.tf._
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.log4j._
import org.opencv.imgproc.Imgproc

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import scala.language.existentials
import scala.reflect.ClassTag


/**
 * [[com.intel.analytics.bigdl.dataset.Sample]] for python.
 * @param features features
 * @param labels labels
 * @param bigdlType bigdl numeric type
 */
case class Sample(features: JList[JTensor],
                  labels: JList[JTensor],
                  bigdlType: String)

case class JTensor(storage: Array[Float], shape: Array[Int],
                   bigdlType: String, indices: Array[Array[Int]] = null)

case class JActivity(value: Activity)

/**
 * [[ValidationResult]] for python
 * @param result result
 * @param totalNum total number
 * @param method method name
 */

case class EvaluatedResult(val result: Float, totalNum: Int, method: String)

object PythonBigDL {

  def ofFloat(): PythonBigDL[Float] = new PythonBigDL[Float]()

  def ofDouble(): PythonBigDL[Double] = new PythonBigDL[Double]()
}

/**
 * Implementation of Python API for BigDL
 */
class PythonBigDL[T: ClassTag](implicit ev: TensorNumeric[T]) extends Serializable {

  private val typeName = {
    val cls = implicitly[ClassTag[T]].runtimeClass
    cls.getSimpleName
  }

  private def toTable(input: JList[_ <: Object]): Table = {
    input.asScala.foldLeft(new Table())((t, e) =>
      if (e.isInstanceOf[JTensor]) {
        t.insert(toTensor(e.asInstanceOf[JTensor]))
      } else {
        t.insert(toTable(e.asInstanceOf[JList[Object]]))
      })
  }

  def jTensorsToActivity(input: JList[_ <: Object], isTable: Boolean): Activity = {
    if (input.isEmpty) {
      throw new IllegalArgumentException("Empty input")
    }
    if (isTable) {
      toTable(input)
    } else {
      toTensor(input.asInstanceOf[JList[JTensor]].iterator().next())
    }
  }

  def activityToJTensors(outputActivity: Activity): JList[JTensor] = {
    if (outputActivity.isInstanceOf[Tensor[T]]) {
      List(toJTensor(outputActivity.toTensor)).asJava
    } else if (outputActivity.isInstanceOf[Table]) {
      outputActivity.toTable.getState().toList.map {
        pair => (pair._1.asInstanceOf[Int], toJTensor(pair._2.asInstanceOf[Tensor[T]]))
      }.sortWith(_._1 < _._1).map(pair => pair._2).asJava
    } else if (outputActivity.isInstanceOf[EmptyGradInput]) {
      List[JTensor]().asJava
    } else {
      throw new UnsupportedOperationException(s"Activity type" +
        s"(${outputActivity.getClass.getName}) not support")
    }
  }

  def toPySample(sample: JSample[T]): Sample = {
    val cls = implicitly[ClassTag[T]].runtimeClass
    val features = new JArrayList[JTensor]()
    features.add(toJTensor(sample.feature()))
    val labels = new JArrayList[JTensor]()
    labels.add(toJTensor(sample.label()))
    Sample(features, labels, cls.getSimpleName)
  }

  def toTensor(jTensor: JTensor): Tensor[T] = {
    if (jTensor == null) return null

    this.typeName match {
      case "float" =>
        if (null == jTensor.indices) {
          if (jTensor.shape == null || jTensor.shape.length == 0) {
            Tensor()
          } else {
            Tensor(jTensor.storage.map(x => ev.fromType(x)), jTensor.shape)
          }
        } else {
          Tensor.sparse(jTensor.indices, jTensor.storage.map(x => ev.fromType(x)), jTensor.shape)
        }
      case "double" =>
        if (null == jTensor.indices) {
          if (jTensor.shape == null || jTensor.shape.length == 0) {
            Tensor()
          } else {
            Tensor(jTensor.storage.map(x => ev.fromType(x.toDouble)), jTensor.shape)
          }
        } else {
          Tensor.sparse(jTensor.indices,
            jTensor.storage.map(x => ev.fromType(x.toDouble)), jTensor.shape)
        }
      case t: String =>
        throw new IllegalArgumentException(s"Not supported type: ${t}")
    }
  }

  def toJTensor(tensor: Tensor[T]): JTensor = {
    // clone here in case the the size of storage larger then the size of tensor.
    require(tensor != null, "tensor cannot be null")
    tensor.getTensorType match {
      case SparseType =>
        // Note: as SparseTensor's indices is inaccessible here,
        // so we will transfer it to DenseTensor. Just for testing.
        if (tensor.nElement() == 0) {
          JTensor(Array(), Array(0), bigdlType = typeName)
        } else {
          val cloneTensor = Tensor.dense(tensor)
          val result = JTensor(cloneTensor.storage().array().map(i => ev.toType[Float](i)),
            cloneTensor.size(), bigdlType = typeName)
          result
        }
      case DenseType =>
        if (tensor.nElement() == 0) {
          if (tensor.dim() == 0) {
            JTensor(null, null, bigdlType = typeName)
          } else {
            JTensor(Array(), tensor.size(), bigdlType = typeName)
          }
        } else {
          val cloneTensor = tensor.clone()
          val result = JTensor(cloneTensor.storage().array().map(i => ev.toType[Float](i)),
            cloneTensor.size(), bigdlType = typeName)
          result
        }
      case _ =>
        throw new IllegalArgumentException(s"toJTensor: Unsupported tensor type" +
          s" ${tensor.getTensorType}")
    }
  }

  def testTensor(jTensor: JTensor): JTensor = {
    val tensor = toTensor(jTensor)
    toJTensor(tensor)
  }


  def testSample(sample: Sample): Sample = {
    val jsample = toJSample(sample)
    toPySample(jsample)
  }

  def toJSample(record: Sample): JSample[T] = {
    require(record.bigdlType == this.typeName,
      s"record.bigdlType: ${record.bigdlType} == this.typeName: ${this.typeName}")
    JSample[T](record.features.asScala.toArray.map(toTensor(_)),
      record.labels.asScala.toArray.map(toTensor(_)))
  }

  def toJSample(psamples: RDD[Sample]): RDD[JSample[T]] = {
    psamples.map(toJSample(_))
  }

  // The first dimension is batch for both X and y
  def toSampleArray(Xs: List[Tensor[T]], y: Tensor[T] = null): Array[JSample[T]] = {
    require(!Xs.isEmpty, "Xs should not be empty")
    val totalNum = Xs(0).size()(0)
    var i = 1
    val samples = new Array[JSample[T]](totalNum)

    if (y != null) {
      require(Xs(0).size()(0) == y.size()(0),
        s"The batch dim should be equal, but we got: ${Xs(0).size()(0)} vs ${y.size()(0)}")
      while (i <= totalNum) {
        samples(i-1) = JSample(Xs.map{X => X.select(1, i)}.toArray, y.select(1, i))
        i += 1
      }
    } else {
      val dummyTensor = Tensor[T](1).fill(ev.fromType(1))
      while (i <= totalNum) {
        samples(i-1) = JSample(Xs.map{X => X.select(1, i)}.toArray, dummyTensor)
        i += 1
      }
    }

    samples
  }


  def batching(dataset: DataSet[JSample[T]], batchSize: Int)
  : DataSet[MiniBatch[T]] = {
    dataset -> SampleToMiniBatch[T](batchSize)
  }

  private def enrichOptimizer[T](
        optimizer: Optimizer[T, MiniBatch[T]],
        endTrigger: Trigger,
        optimMethod: Map[String, OptimMethod[T]]): Optimizer[T, MiniBatch[T]] = {
    optimizer.setEndWhen(endTrigger)

    optimizer.setOptimMethods(optimMethod)

    // TODO: remove this
    optimizer.disableCheckSingleton()

    optimizer
  }

  def createSequential(): Container[Activity, Activity, T] = {
      Sequential[T]()
  }

  def toGraph(sequential: Sequential[T]): StaticGraph[T] = {
    sequential.toGraph().asInstanceOf[StaticGraph[T]]
  }

  def createAttention(hiddenSize: Int, numHeads: Int, attentionDropout: Float): Attention[T] = {
     Attention(hiddenSize, numHeads, attentionDropout)
  }

  def createFeedForwardNetwork(hiddenSize: Int,
    filterSize: Int, reluDropout: Float): FeedForwardNetwork[T] = {
    FeedForwardNetwork(hiddenSize, filterSize, reluDropout)
  }

  def createExpandSize(targetSizes: JList[Int]): ExpandSize[T] = {
    ExpandSize(targetSizes.asScala.toArray)
  }

  def createTableOperation(
    operationLayer: AbstractModule[Table, Tensor[T], T]): TableOperation[T] = {
    new TableOperation(operationLayer)
  }

  def createLayerNormalization(hiddenSize: Int): LayerNormalization[T] = {
    new LayerNormalization[T](hiddenSize)
  }

  def createTransformer(
    vocabSize: Int,
    hiddenSize: Int,
    numHeads: Int,
    filterSize: Int,
    numHiddenlayers: Int,
    postprocessDropout: Double,
    attentionDropout: Double,
    reluDropout: Double): nn.Transformer[T] = {
    Transformer(vocabSize, hiddenSize, numHeads,
      filterSize, numHiddenlayers, postprocessDropout.toFloat,
      attentionDropout.toFloat, reluDropout.toFloat)
  }

  def createLinear(inputSize: Int, outputSize: Int,
    withBias: Boolean,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    initWeight: JTensor = null,
    initBias: JTensor = null,
    initGradWeight: JTensor = null,
    initGradBias: JTensor = null): Linear[T] = {
    Linear[T](inputSize, outputSize, withBias, wRegularizer, bRegularizer,
      toTensor(initWeight), toTensor(initBias), toTensor(initGradWeight), toTensor(initGradBias))
  }

  def createSparseLinear(inputSize: Int, outputSize: Int,
                   withBias: Boolean,
                   backwardStart: Int = -1,
                   backwardLength: Int = -1,
                   wRegularizer: Regularizer[T] = null,
                   bRegularizer: Regularizer[T] = null,
                   initWeight: JTensor = null,
                   initBias: JTensor = null,
                   initGradWeight: JTensor = null,
                   initGradBias: JTensor = null): SparseLinear[T] = {
    SparseLinear[T](inputSize, outputSize, withBias, backwardStart, backwardLength,
      wRegularizer, bRegularizer, toTensor(initWeight), toTensor(initBias),
      toTensor(initGradWeight), toTensor(initGradBias))
  }

  def createNegative(inplace: Boolean): Negative[T] = {
    Negative[T](inplace)
  }

  def createDenseToSparse(): DenseToSparse[T] = {
    DenseToSparse[T]()
  }

  def createReLU(ip: Boolean = false): ReLU[T] = {
    ReLU[T](ip)
  }

  def createTanh(): Tanh[T] = {
    Tanh[T]()
  }

  def createTimeDistributed(layer: TensorModule[T]): TimeDistributed[T] = {
    TimeDistributed[T](layer)
  }

  def createSpatialWithinChannelLRN(size: Int = 5, alpha: Double = 1.0, beta: Double = 0.75)
  : SpatialWithinChannelLRN[T] = {
    SpatialWithinChannelLRN[T](size, alpha, beta)
  }

  def createRnnCell(inputSize: Int,
    hiddenSize: Int,
    activation: TensorModule[T],
    isInputWithBias: Boolean = true,
    isHiddenWithBias: Boolean = true,
    wRegularizer: Regularizer[T] = null,
    uRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null): RnnCell[T] = {
    RnnCell[T](inputSize,
      hiddenSize,
      activation,
      isInputWithBias,
      isHiddenWithBias,
      wRegularizer,
      uRegularizer,
      bRegularizer)
  }

  def createTimeDistributedMaskCriterion(critrn: TensorCriterion[T],
    paddingValue: Int = 0): TimeDistributedMaskCriterion[T] = {
    TimeDistributedMaskCriterion[T](critrn, paddingValue)
  }

  def createTimeDistributedCriterion(critrn: TensorCriterion[T],
    sizeAverage: Boolean = false, dimension: Int = 2): TimeDistributedCriterion[T] = {
    TimeDistributedCriterion[T](critrn, sizeAverage, dimension)
  }

  def createGRU(
    inputSize: Int,
    outputSize: Int,
    p: Double = 0,
    activation: TensorModule[T] = null,
    innerActivation: TensorModule[T] = null,
    wRegularizer: Regularizer[T] = null,
    uRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null): GRU[T] = {
    GRU[T](inputSize, outputSize, p, activation, innerActivation,
      wRegularizer, uRegularizer, bRegularizer)
  }

  def createLSTM(
    inputSize: Int,
    hiddenSize: Int,
    p: Double = 0,
    activation: TensorModule[T] = null,
    innerActivation: TensorModule[T] = null,
    wRegularizer: Regularizer[T] = null,
    uRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null): LSTM[T] = {
    LSTM[T](inputSize, hiddenSize, p, activation, innerActivation,
      wRegularizer, uRegularizer, bRegularizer)
  }

  def createLSTMPeephole(
    inputSize: Int,
    hiddenSize: Int,
    p: Double = 0,
    wRegularizer: Regularizer[T] = null,
    uRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null): LSTMPeephole[T] = {
    LSTMPeephole[T](inputSize, hiddenSize, p, wRegularizer, uRegularizer, bRegularizer)
  }

  def createRecurrent(): Recurrent[T] = {
    Recurrent[T]()
  }

  def createRecurrentDecoder(outputLength: Int): RecurrentDecoder[T] = {
    RecurrentDecoder[T](outputLength)
  }

  def createConvLSTMPeephole(
    inputSize: Int,
    outputSize: Int,
    kernelI: Int,
    kernelC: Int,
    stride: Int = 1,
    padding: Int = -1,
    activation: TensorModule[T] = null,
    innerActivation: TensorModule[T] = null,
    wRegularizer: Regularizer[T] = null,
    uRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    cRegularizer: Regularizer[T] = null,
    withPeephole: Boolean = true): ConvLSTMPeephole[T] = {
    ConvLSTMPeephole[T](inputSize, outputSize, kernelI, kernelC,
      stride, padding, activation, innerActivation,
      wRegularizer, uRegularizer, bRegularizer, cRegularizer, withPeephole)
  }

  def createConvLSTMPeephole3D(
    inputSize: Int,
    outputSize: Int,
    kernelI: Int,
    kernelC: Int,
    stride: Int = 1,
    padding: Int = -1,
    wRegularizer: Regularizer[T] = null,
    uRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    cRegularizer: Regularizer[T] = null,
    withPeephole: Boolean = true): ConvLSTMPeephole3D[T] = {
    ConvLSTMPeephole3D[T](inputSize, outputSize, kernelI, kernelC, stride, padding,
      wRegularizer, uRegularizer, bRegularizer, cRegularizer, withPeephole)
  }

  def createEcho(): Echo[T] = {
    Echo[T]()
  }

  def createLogSoftMax(): LogSoftMax[T] = {
    LogSoftMax[T]()
  }

  def createTemporalMaxPooling(
    kW: Int,
    dW: Int)
  : TemporalMaxPooling[T] = {
    TemporalMaxPooling[T](
      kW,
      dW)
  }

  def createSpatialMaxPooling(kW: Int,
    kH: Int,
    dW: Int,
    dH: Int,
    padW: Int = 0,
    padH: Int = 0,
    ceilMode: Boolean = false,
    format: String = "NCHW")
  : SpatialMaxPooling[T] = {
    val maxpooling = SpatialMaxPooling[T](kW,
      kH,
      dW,
      dH,
      padW,
      padH,
      format = DataFormat(format))
    if (ceilMode) maxpooling.ceil()
    else maxpooling
  }

  def createLocallyConnected2D(
    nInputPlane: Int,
    inputWidth: Int,
    inputHeight: Int,
    nOutputPlane: Int,
    kernelW: Int,
    kernelH: Int,
    strideW: Int = 1,
    strideH: Int = 1,
    padW: Int = 0,
    padH: Int = 0,
    propagateBack: Boolean = true,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    initWeight: JTensor = null,
    initBias: JTensor = null,
    initGradWeight: JTensor = null,
    initGradBias: JTensor = null,
    withBias: Boolean = true,
    dataFormat: String = "NCHW"): LocallyConnected2D[T] = {
    LocallyConnected2D[T](
      nInputPlane,
      inputWidth,
      inputHeight,
      nOutputPlane,
      kernelW,
      kernelH,
      strideW,
      strideH,
      padW,
      padH,
      propagateBack,
      wRegularizer,
      bRegularizer,
      toTensor(initWeight),
      toTensor(initBias),
      toTensor(initGradWeight),
      toTensor(initGradBias),
      withBias,
      DataFormat(dataFormat)
    )
  }

  def createSpatialConvolution(nInputPlane: Int,
    nOutputPlane: Int,
    kernelW: Int,
    kernelH: Int,
    strideW: Int = 1,
    strideH: Int = 1,
    padW: Int = 0,
    padH: Int = 0,
    nGroup: Int = 1,
    propagateBack: Boolean = true,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    initWeight: JTensor = null,
    initBias: JTensor = null,
    initGradWeight: JTensor = null,
    initGradBias: JTensor = null,
    withBias: Boolean = true,
    dataFormat: String = "NCHW"
  )
  : SpatialConvolution[T] = {
    SpatialConvolution[T](nInputPlane,
      nOutputPlane,
      kernelW,
      kernelH,
      strideW,
      strideH,
      padW,
      padH,
      nGroup,
      propagateBack,
      wRegularizer,
      bRegularizer,
      toTensor(initWeight),
      toTensor(initBias),
      toTensor(initGradWeight),
      toTensor(initGradBias),
      withBias,
      DataFormat(dataFormat)
    )
  }

  def createSpatialSeparableConvolution(
    nInputChannel: Int,
    nOutputChannel: Int,
    depthMultiplier: Int,
    kW: Int,
    kH: Int,
    sW: Int = 1,
    sH: Int = 1,
    pW: Int = 0,
    pH: Int = 0,
    withBias: Boolean = true,
    dataFormat: String = "NCHW",
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    pRegularizer: Regularizer[T] = null
  )
  : SpatialSeparableConvolution[T] = {
    SpatialSeparableConvolution[T](nInputChannel,
      nOutputChannel,
      depthMultiplier,
      kW,
      kH,
      sW,
      sH,
      pW,
      pH,
      withBias,
      DataFormat(dataFormat),
      wRegularizer,
      bRegularizer,
      pRegularizer
    )
  }

  def createReshape(size: JList[Int], batchMode: JBoolean = null): Reshape[T] = {
    val mappedBatchMode = batchMode match {
      case JBoolean.TRUE => Some(true)
      case JBoolean.FALSE => Some(false)
      case _ => None
    }
    Reshape(size.asScala.toArray, mappedBatchMode)
  }

  def createConcat(dimension: Int): Concat[T] = {
    Concat[T](dimension)
  }

  def createSpatialAveragePooling(kW: Int,
    kH: Int,
    dW: Int = 1,
    dH: Int = 1,
    padW: Int = 0,
    padH: Int = 0,
    globalPooling: Boolean = false,
    ceilMode: Boolean = false,
    countIncludePad: Boolean = true,
    divide: Boolean = true,
    format: String = "NCHW")
  : SpatialAveragePooling[T] = {
    SpatialAveragePooling[T](kW, kH, dW, dH, padW, padH, globalPooling,
      ceilMode, countIncludePad, divide, format = DataFormat(format))
  }

  def createSpatialBatchNormalization(nOutput: Int,
    eps: Double = 1e-5,
    momentum: Double = 0.1,
    affine: Boolean = true,
    initWeight: JTensor = null,
    initBias: JTensor = null,
    initGradWeight: JTensor = null,
    initGradBias: JTensor = null, dataFormat: String = "NCHW")
  : SpatialBatchNormalization[T] = {
    SpatialBatchNormalization[T](nOutput, eps, momentum, affine,
      toTensor(initWeight), toTensor(initBias), toTensor(initGradWeight), toTensor(initBias),
      DataFormat(dataFormat)
    )
  }

  def createSpatialCrossMapLRN(size: Int = 5,
    alpha: Double = 1.0,
    beta: Double = 0.75,
    k: Double = 1.0,
    dataFormat: String = "NCHW")
  : SpatialCrossMapLRN[T] = {
    SpatialCrossMapLRN[T](size, alpha, beta, k, DataFormat(dataFormat))
  }

  def createDropout(initP: Double = 0.5,
    inplace: Boolean = false,
    scale: Boolean = true)
  : Dropout[T] = {
    Dropout[T](initP, inplace, scale)
  }

  def createGaussianDropout(rate: Double)
  : GaussianDropout[T] = {
    GaussianDropout[T](rate)
  }

  def createGaussianNoise(stddev: Double)
  : GaussianNoise[T] = {
    GaussianNoise[T](stddev)
  }

  def createView(sizes: JList[Int], num_input_dims: Int = 0): View[T] = {
    View[T](sizes.asScala.toArray).setNumInputDims(num_input_dims)
  }

  def createAbs()
  : Abs[T] = {
    Abs[T]()
  }

  def createAdd(inputSize: Int)
  : Add[T] = {
    Add[T](inputSize)
  }

  def createAddConstant(constant_scalar: Double,
    inplace: Boolean = false)
  : AddConstant[T] = {
    AddConstant[T](constant_scalar,
      inplace)
  }


  def createBatchNormalization(nOutput: Int,
    eps: Double = 1e-5,
    momentum: Double = 0.1,
    affine: Boolean = true,
    initWeight: JTensor = null,
    initBias: JTensor = null,
    initGradWeight: JTensor = null,
    initGradBias: JTensor = null)
  : BatchNormalization[T] = {
    BatchNormalization[T](nOutput,
      eps,
      momentum,
      affine,
      toTensor(initWeight),
      toTensor(initBias),
      toTensor(initGradWeight),
      toTensor(initGradBias))
  }

  def createBilinear(inputSize1: Int,
    inputSize2: Int,
    outputSize: Int,
    biasRes: Boolean = true,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null)
  : Bilinear[T] = {
    Bilinear[T](inputSize1,
      inputSize2,
      outputSize,
      biasRes,
      wRegularizer,
      bRegularizer)
  }

  def createBottle(module: AbstractModule[Activity, Activity, T],
    nInputDim: Int = 2,
    nOutputDim1: Int = Int.MaxValue)
  : Bottle[T] = {
    Bottle[T](module,
      nInputDim,
      nOutputDim1)
  }

  def createCAdd(size: JList[Int],
    bRegularizer: Regularizer[T] = null)
  : CAdd[T] = {
    CAdd[T](size.asScala.toArray, bRegularizer)
  }

  def createCAddTable(inplace: Boolean = false)
  : CAddTable[T, T] = {
    CAddTable[T](inplace)
  }

  def createCAveTable(inplace: Boolean = false)
  : CAveTable[T] = {
    CAveTable[T](inplace)
  }

  def createCDivTable()
  : CDivTable[T] = {
    CDivTable[T]()
  }

  def createCMaxTable()
  : CMaxTable[T] = {
    CMaxTable[T]()
  }

  def createCMinTable()
  : CMinTable[T] = {
    CMinTable[T]()
  }

  def createCMul(size: JList[Int],
    wRegularizer: Regularizer[T] = null)
  : CMul[T] = {
    CMul[T](size.asScala.toArray, wRegularizer)
  }

  def createCMulTable()
  : CMulTable[T] = {
    CMulTable[T]()
  }

  def createCSubTable()
  : CSubTable[T] = {
    CSubTable[T]()
  }

  def createClamp(min: Int,
    max: Int)
  : Clamp[T] = {
    Clamp[T](min,
      max)
  }

  def createContiguous()
  : Contiguous[T] = {
    Contiguous[T]()
  }

  def createCosine(inputSize: Int,
    outputSize: Int)
  : Cosine[T] = {
    Cosine[T](inputSize,
      outputSize)
  }

  def createCosineDistance()
  : CosineDistance[T] = {
    CosineDistance[T]()
  }

  def createCosineDistanceCriterion(sizeAverage: Boolean = true)
  : CosineDistanceCriterion[T] = {
    CosineDistanceCriterion[T](sizeAverage)
  }

  def createCrossProduct(numTensor: Int = 0,
    embeddingSize: Int = 0)
  : CrossProduct[T] = {
    CrossProduct[T](numTensor, embeddingSize)
  }

  def createDiceCoefficientCriterion(sizeAverage: Boolean = true,
    epsilon: Float = 1.0f)
  : DiceCoefficientCriterion[T] = {
    DiceCoefficientCriterion[T](sizeAverage, epsilon)
  }

  def createDotProduct()
  : DotProduct[T] = {
    DotProduct[T]()
  }

  def createELU(alpha: Double = 1.0,
    inplace: Boolean = false)
  : ELU[T] = {
    ELU[T](alpha,
      inplace)
  }

  def createEuclidean(inputSize: Int,
    outputSize: Int,
    fastBackward: Boolean = true)
  : Euclidean[T] = {
    Euclidean[T](inputSize,
      outputSize,
      fastBackward)
  }

  def createExp()
  : Exp[T] = {
    Exp[T]()
  }

  def createFlattenTable()
  : FlattenTable[T] = {
    FlattenTable[T]()
  }

  def createGradientReversal(lambda: Double = 1)
  : GradientReversal[T] = {
    GradientReversal[T](lambda)
  }

  def createHardShrink(lambda: Double = 0.5)
  : HardShrink[T] = {
    HardShrink[T](lambda)
  }

  def createHardTanh(minValue: Double = -1,
    maxValue: Double = 1,
    inplace: Boolean = false)
  : HardTanh[T] = {
    HardTanh[T](minValue,
      maxValue,
      inplace)
  }

  def createIndex(dimension: Int)
  : Index[T] = {
    Index[T](dimension)
  }

  def createInferReshape(size: JList[Int], batchMode: Boolean = false)
  : InferReshape[T] = {
    InferReshape[T](size.asScala.toArray,
      batchMode)
  }

  def createJoinTable(dimension: Int,
    nInputDims: Int)
  : JoinTable[T] = {
    JoinTable[T](dimension,
      nInputDims)
  }

  def createSparseJoinTable(dimension: Int): SparseJoinTable[T] = {
    SparseJoinTable[T](dimension)
  }

  def createL1Cost()
  : L1Cost[T] = {
    L1Cost[T]()
  }

  def createUpSampling1D(length: Int): UpSampling1D[T] = {
    UpSampling1D(length)
  }

  def createUpSampling2D(size: JList[Int], dataFormat: String): UpSampling2D[T] = {
    UpSampling2D(size.asScala.toArray, DataFormat(dataFormat))
  }

  def createL1Penalty(l1weight: Int,
    sizeAverage: Boolean = false,
    provideOutput: Boolean = true)
  : L1Penalty[T] = {
    L1Penalty[T](l1weight,
      sizeAverage,
      provideOutput)
  }

  def createNegativeEntropyPenalty(beta: Double): NegativeEntropyPenalty[T] = {
    NegativeEntropyPenalty(beta)
  }

  def createLeakyReLU(negval: Double = 0.01,
    inplace: Boolean = false)
  : LeakyReLU[T] = {
    LeakyReLU[T](negval,
      inplace)
  }

  def createLog()
  : Log[T] = {
    Log[T]()
  }

  def createLogSigmoid()
  : LogSigmoid[T] = {
    LogSigmoid[T]()
  }

  def createLookupTable(nIndex: Int, nOutput: Int,
    paddingValue: Double = 0, maxNorm: Double = Double.MaxValue,
    normType: Double = 2.0, shouldScaleGradByFreq: Boolean = false,
    wRegularizer: Regularizer[T] = null)
  : LookupTable[T] = {
    LookupTable[T](nIndex,
      nOutput,
      paddingValue,
      maxNorm,
      normType,
      shouldScaleGradByFreq,
      wRegularizer)
  }

  def createLookupTableSparse(nIndex: Int, nOutput: Int,
    combiner: String = "sum", maxNorm: Double = -1,
    wRegularizer: Regularizer[T] = null)
  : LookupTableSparse[T] = {
    LookupTableSparse[T](nIndex,
      nOutput,
      combiner,
      maxNorm,
      wRegularizer)
  }

  def createMM(transA: Boolean = false,
    transB: Boolean = false)
  : MM[T] = {
    MM[T](transA,
      transB)
  }

  def createMV(trans: Boolean = false)
  : MV[T] = {
    MV[T](trans)
  }

  def createMapTable(module: AbstractModule[Activity, Activity, T] = null)
  : MapTable[T] = {
    MapTable[T](module)
  }

  def createMaskedSelect()
  : MaskedSelect[T] = {
    MaskedSelect[T]()
  }

  def createMax(dim: Int = 1,
    numInputDims: Int = Int.MinValue)
  : Max[T] = {
    Max[T](dim,
      numInputDims)
  }

  def createMean(dimension: Int = 1,
    nInputDims: Int = -1,
    squeeze: Boolean = true)
  : Mean[T] = {
    Mean[T](dimension,
      nInputDims,
      squeeze)
  }

  def createMin(dim: Int = 1,
    numInputDims: Int = Int.MinValue)
  : Min[T] = {
    Min[T](dim,
      numInputDims)
  }

  def createMixtureTable(dim: Int = Int.MaxValue)
  : MixtureTable[T] = {
    MixtureTable[T](dim)
  }

  def createMul()
  : Mul[T] = {
    Mul[T]()
  }

  def createMulConstant(scalar: Double,
    inplace: Boolean = false)
  : MulConstant[T] = {
    MulConstant[T](scalar,
      inplace)
  }

  def createNarrow(dimension: Int,
    offset: Int,
    length: Int = 1)
  : Narrow[T] = {
    Narrow[T](dimension,
      offset,
      length)
  }

  def createNarrowTable(offset: Int,
    length: Int = 1)
  : NarrowTable[T] = {
    NarrowTable[T](offset,
      length)
  }

  def createNormalize(p: Double,
    eps: Double = 1e-10)
  : Normalize[T] = {
    Normalize[T](p,
      eps)
  }

  def createPReLU(nOutputPlane: Int = 0)
  : PReLU[T] = {
    PReLU[T](nOutputPlane)
  }

  def createSReLU(shape: JArrayList[Int], shareAxes: JArrayList[Int] = null): SReLU[T] = {
    val argv: Array[Int] = if (shareAxes == null) {
      null
    } else {
      shareAxes.asScala.toArray
    }
    SReLU[T](shape.asScala.toArray, argv)
  }

  def createActivityRegularization(l1: Double, l2: Double): ActivityRegularization[T] = {
    ActivityRegularization[T](l1, l2)
  }

  def createPadding(dim: Int,
    pad: Int,
    nInputDim: Int,
    value: Double = 0.0,
    nIndex: Int = 1)
  : Padding[T] = {
    Padding[T](dim,
      pad,
      nInputDim,
      value,
      nIndex)
  }

  def createPairwiseDistance(norm: Int = 2)
  : PairwiseDistance[T] = {
    PairwiseDistance[T](norm)
  }

  def createParallelTable()
  : ParallelTable[T] = {
    ParallelTable[T]()
  }

  def createPower(power: Double,
    scale: Double = 1,
    shift: Double = 0)
  : Power[T] = {
    Power[T](power,
      scale,
      shift)
  }

  def createRReLU(lower: Double = 1.0 / 8,
    upper: Double = 1.0 / 3,
    inplace: Boolean = false)
  : RReLU[T] = {
    RReLU[T](lower,
      upper,
      inplace)
  }

  def createReLU6(inplace: Boolean = false)
  : ReLU6[T] = {
    ReLU6[T](inplace)
  }

  def createReplicate(nFeatures: Int,
    dim: Int = 1,
    nDim: Int = Int.MaxValue)
  : Replicate[T] = {
    Replicate[T](nFeatures,
      dim,
      nDim)
  }

  def createRoiPooling(pooled_w: Int, pooled_h: Int, spatial_scale: Double)
  : RoiPooling[T] = {
    RoiPooling[T](pooled_w,
      pooled_h,
      ev.fromType(spatial_scale))
  }

  def createRoiAlign(spatial_scale: Double, sampling_ratio: Int, pooled_h: Int, pooled_w: Int)
  : RoiAlign[T] = {
    RoiAlign[T](spatial_scale.toFloat,
      sampling_ratio,
      pooled_h,
      pooled_w)
  }

  def createFPN(in_channels_list: JList[Int], out_channels: Int,
                top_blocks: Int = 0, in_channels_of_p6p7: Int = 0, out_channels_of_p6p7: Int = 0)
  : FPN[T] = {
    FPN[T](in_channels_list.asScala.toArray, out_channels,
      top_blocks, in_channels_of_p6p7, out_channels_of_p6p7)
  }

  def createPooler(resolution: Int, scales: JList[Double], sampling_ratio: Int)
  : Pooler[T] = {
    Pooler[T](resolution,
      scales.asScala.toArray.map(_.toFloat),
      sampling_ratio)
  }

  def createScale(size: JList[Int])
  : Scale[T] = {
    Scale[T](size.asScala.toArray)
  }

  def createSelect(dimension: Int,
    index: Int)
  : Select[T] = {
    Select[T](dimension,
      index)
  }

  def createSelectTable(dimension: Int)
  : SelectTable[T] = {
    SelectTable[T](dimension)
  }

  def createSequenceBeamSearch(vocabSize: Int,
    beamSize: Int,
    alpha: Float,
    decodeLength: Int,
    eosId: Float,
    paddingValue: Float,
    numHiddenLayers: Int,
    hiddenSize: Int)
  : SequenceBeamSearch[T] = {
    SequenceBeamSearch[T](vocabSize,
      beamSize,
      alpha,
      decodeLength,
      eosId,
      paddingValue,
      numHiddenLayers,
      hiddenSize)
  }

  def createSigmoid()
  : Sigmoid[T] = {
    Sigmoid[T]()
  }

  def createSoftMax(pos: Int = 1)
  : SoftMax[T] = {
    SoftMax[T](pos)
  }

  def createSoftMin()
  : SoftMin[T] = {
    SoftMin[T]()
  }

  def createSoftPlus(beta: Double = 1.0)
  : SoftPlus[T] = {
    SoftPlus[T](beta)
  }

  def createSoftShrink(lambda: Double = 0.5)
  : SoftShrink[T] = {
    SoftShrink[T](lambda)
  }

  def createSoftSign()
  : SoftSign[T] = {
    SoftSign[T]()
  }


  def createSpatialDropout1D(
    initP: Double = 0.5
  ): SpatialDropout1D[T] = {
    SpatialDropout1D[T](initP)
  }

  def createSpatialDropout2D(
    initP: Double = 0.5,
    dataFormat: String = "NCHW"
  ): SpatialDropout2D[T] = {
    SpatialDropout2D[T](initP, DataFormat(dataFormat))
  }

  def createSpatialDropout3D(
    initP: Double = 0.5,
    dataFormat: String = "NCHW"
  ): SpatialDropout3D[T] = {
    SpatialDropout3D[T](initP, DataFormat(dataFormat))
  }

  def createSpatialDilatedConvolution(nInputPlane: Int,
    nOutputPlane: Int,
    kW: Int,
    kH: Int,
    dW: Int = 1,
    dH: Int = 1,
    padW: Int = 0,
    padH: Int = 0,
    dilationW: Int = 1,
    dilationH: Int = 1,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null)
  : SpatialDilatedConvolution[T] = {
    SpatialDilatedConvolution[T](nInputPlane,
      nOutputPlane,
      kW,
      kH,
      dW,
      dH,
      padW,
      padH,
      dilationW,
      dilationH,
      wRegularizer,
      bRegularizer)
  }

  def createTemporalConvolution(
    inputFrameSize: Int,
    outputFrameSize: Int,
    kernelW: Int,
    strideW: Int = 1,
    propagateBack: Boolean = true,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    initWeight: JTensor = null,
    initBias: JTensor = null,
    initGradWeight: JTensor = null,
    initGradBias: JTensor = null
  )
  : TemporalConvolution[T] = {
    TemporalConvolution[T](
      inputFrameSize,
      outputFrameSize,
      kernelW,
      strideW,
      propagateBack,
      wRegularizer,
      bRegularizer,
      toTensor(initWeight),
      toTensor(initBias),
      toTensor(initGradWeight),
      toTensor(initGradBias)
    )
  }

  def createLocallyConnected1D(
                                nInputFrame: Int,
                                inputFrameSize: Int,
                                outputFrameSize: Int,
                                kernelW: Int,
                                strideW: Int = 1,
                                propagateBack: Boolean = true,
                                wRegularizer: Regularizer[T] = null,
                                bRegularizer: Regularizer[T] = null,
                                initWeight: JTensor = null,
                                initBias: JTensor = null,
                                initGradWeight: JTensor = null,
                                initGradBias: JTensor = null
                              )
  : LocallyConnected1D[T] = {
    LocallyConnected1D[T](
      nInputFrame,
      inputFrameSize,
      outputFrameSize,
      kernelW,
      strideW,
      propagateBack,
      wRegularizer,
      bRegularizer,
      toTensor(initWeight),
      toTensor(initBias),
      toTensor(initGradWeight),
      toTensor(initGradBias)
    )
  }

  def createBinaryTreeLSTM(
    inputSize: Int,
    hiddenSize: Int,
    gateOutput: Boolean = true,
    withGraph: Boolean = true)
  : BinaryTreeLSTM[T] = {
    BinaryTreeLSTM[T](
      inputSize,
      hiddenSize,
      gateOutput,
      withGraph)
  }

  def createVolumetricFullConvolution(nInputPlane: Int,
    nOutputPlane: Int,
    kT: Int,
    kW: Int,
    kH: Int,
    dT: Int = 1,
    dW: Int = 1,
    dH: Int = 1,
    padT: Int = 0,
    padW: Int = 0,
    padH: Int = 0,
    adjT: Int = 0,
    adjW: Int = 0,
    adjH: Int = 0,
    nGroup: Int = 1,
    noBias: Boolean = false,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null)
  : VolumetricFullConvolution[T] = {
    VolumetricFullConvolution[T](nInputPlane,
      nOutputPlane,
      kT,
      kW,
      kH,
      dT,
      dW,
      dH,
      padT,
      padW,
      padH,
      adjT,
      adjW,
      adjH,
      nGroup,
      noBias,
      wRegularizer,
      bRegularizer)
  }

  def createSpatialFullConvolution(nInputPlane: Int,
    nOutputPlane: Int,
    kW: Int,
    kH: Int,
    dW: Int = 1,
    dH: Int = 1,
    padW: Int = 0,
    padH: Int = 0,
    adjW: Int = 0,
    adjH: Int = 0,
    nGroup: Int = 1,
    noBias: Boolean = false,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null)
  : SpatialFullConvolution[T] = {
    SpatialFullConvolution[T](nInputPlane,
      nOutputPlane,
      kW,
      kH,
      dW,
      dH,
      padW,
      padH,
      adjW,
      adjH,
      nGroup,
      noBias,
      wRegularizer,
      bRegularizer)
  }

  def createSpatialShareConvolution(
    nInputPlane: Int,
    nOutputPlane: Int,
    kernelW: Int,
    kernelH: Int,
    strideW: Int = 1,
    strideH: Int = 1,
    padW: Int = 0,
    padH: Int = 0,
    nGroup: Int = 1,
    propagateBack: Boolean = true,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    initWeight: JTensor = null,
    initBias: JTensor = null,
    initGradWeight: JTensor = null,
    initGradBias: JTensor = null,
    withBias: Boolean = true): SpatialShareConvolution[T] = {
    SpatialShareConvolution[T](nInputPlane,
      nOutputPlane,
      kernelW,
      kernelH,
      strideW,
      strideH,
      padW,
      padH,
      nGroup,
      propagateBack,
      wRegularizer,
      bRegularizer,
      toTensor(initWeight),
      toTensor(initBias),
      toTensor(initGradWeight),
      toTensor(initGradBias),
      withBias
    )
  }

  def createSpatialZeroPadding(padLeft: Int,
    padRight: Int,
    padTop: Int,
    padBottom: Int)
  : SpatialZeroPadding[T] = {
    SpatialZeroPadding[T](padLeft,
      padRight,
      padTop,
      padBottom)
  }

  def createBifurcateSplitTable(dimension: Int)
  : BifurcateSplitTable[T] = {
    BifurcateSplitTable[T](dimension)
  }

  def createSplitTable(dimension: Int,
    nInputDims: Int = -1)
  : SplitTable[T] = {
    SplitTable[T](dimension,
      nInputDims)
  }

  def createSqrt()
  : Sqrt[T] = {
    Sqrt[T]()
  }

  def createSquare()
  : Square[T] = {
    Square[T]()
  }

  def createSqueeze(dim: Int = Int.MinValue,
    numInputDims: Int = Int.MinValue)
  : Squeeze[T] = {
    Squeeze[T](dim,
      numInputDims)
  }

  def createSum(dimension: Int = 1,
    nInputDims: Int = -1,
    sizeAverage: Boolean = false,
    squeeze: Boolean = true
  )
  : Sum[T] = {
    Sum[T](dimension,
      nInputDims,
      sizeAverage,
      squeeze
    )
  }

  def createTanhShrink()
  : TanhShrink[T] = {
    TanhShrink[T]()
  }

  def createThreshold(th: Double = 1e-6,
    v: Double = 0.0,
    ip: Boolean = false)
  : Threshold[T] = {
    Threshold[T](th,
      v,
      ip)
  }

  def createUnsqueeze(pos: JList[Int],
    numInputDims: Int = Int.MinValue)
  : Unsqueeze[T] = {
    Unsqueeze[T](pos.asScala.toArray,
      numInputDims)
  }

  def createBCECriterion(weights: JTensor = null,
    sizeAverage: Boolean = true)
  : BCECriterion[T] = {
    BCECriterion[T](if (weights == null) null else toTensor(weights),
      sizeAverage)
  }

  def createBiRecurrent(merge: AbstractModule[Table, Tensor[T], T] = null)
  : BiRecurrent[T] = {
    BiRecurrent[T](merge)
  }

  def createConcatTable()
  : ConcatTable[T] = {
    ConcatTable[Activity, T]()
  }

  def createIdentity()
  : Identity[T] = {
    Identity[T]()
  }

  def createGaussianSampler(): GaussianSampler[T] = {
    GaussianSampler[T]()
  }

  def createMultiLabelSoftMarginCriterion(weights: JTensor = null,
    sizeAverage: Boolean = true)
  : MultiLabelSoftMarginCriterion[T] = {
    MultiLabelSoftMarginCriterion[T](if (weights == null) null else toTensor(weights),
      sizeAverage)
  }

  def createMultiMarginCriterion(p: Int = 1,
    weights: JTensor = null,
    margin: Double = 1.0,
    sizeAverage: Boolean = true)
  : MultiMarginCriterion[T] = {
    MultiMarginCriterion[T](p,
      if (weights == null) null else toTensor(weights),
      margin,
      sizeAverage)
  }

  def createReverse(dimension: Int = 1, isInplace: Boolean = false)
  : Reverse[T] = {
    Reverse[T](dimension, isInplace)
  }

  def createTranspose(permutations: JList[JList[Int]])
  : Transpose[T] = {
    Transpose[T](permutations.asScala.toArray.map { item =>
      val itemArray = item.asScala.toArray
      (itemArray(0), itemArray(1))
    })
  }

  def createSpatialContrastiveNormalization(nInputPlane: Int = 1,
    kernel: JTensor = null,
    threshold: Double = 1e-4,
    thresval: Double = 1e-4)
  : SpatialContrastiveNormalization[T] = {
    SpatialContrastiveNormalization[T](nInputPlane,
      if (kernel == null) null else toTensor(kernel),
      threshold,
      thresval)
  }

  def createSpatialConvolutionMap(connTable: JTensor,
    kW: Int,
    kH: Int,
    dW: Int = 1,
    dH: Int = 1,
    padW: Int = 0,
    padH: Int = 0,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null)
  : SpatialConvolutionMap[T] = {
    SpatialConvolutionMap[T](if (connTable == null) null else toTensor(connTable),
      kW,
      kH,
      dW,
      dH,
      padW,
      padH,
      wRegularizer,
      bRegularizer)
  }

  def createVolumetricConvolution(nInputPlane: Int,
    nOutputPlane: Int,
    kT: Int,
    kW: Int,
    kH: Int,
    dT: Int = 1,
    dW: Int = 1,
    dH: Int = 1,
    padT: Int = 0,
    padW: Int = 0,
    padH: Int = 0,
    withBias: Boolean = true,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null)
  : VolumetricConvolution[T] = {
    VolumetricConvolution[T](nInputPlane,
      nOutputPlane,
      kT,
      kW,
      kH,
      dT,
      dW,
      dH,
      padT,
      padW,
      padH,
      withBias,
      wRegularizer,
      bRegularizer)
  }

  def createVolumetricMaxPooling(kT: Int,
    kW: Int,
    kH: Int,
    dT: Int,
    dW: Int,
    dH: Int,
    padT: Int = 0,
    padW: Int = 0,
    padH: Int = 0): VolumetricMaxPooling[T] = {
    VolumetricMaxPooling[T](kT, kW, kH, dT, dW, dH, padT, padW, padH)
  }

  def createVolumetricAveragePooling(kT: Int,
                                 kW: Int,
                                 kH: Int,
                                 dT: Int,
                                 dW: Int,
                                 dH: Int,
                                 padT: Int = 0,
                                 padW: Int = 0,
                                 padH: Int = 0,
                                 countIncludePad: Boolean = true,
                                 ceilMode: Boolean = false):
  VolumetricAveragePooling[T] = {
    VolumetricAveragePooling[T](kT, kW, kH, dT, dW, dH, padT, padW, padH, countIncludePad, ceilMode)
  }

  def createSpatialDivisiveNormalization(nInputPlane: Int = 1,
    kernel: JTensor = null,
    threshold: Double = 1e-4,
    thresval: Double = 1e-4)
  : SpatialDivisiveNormalization[T] = {
    SpatialDivisiveNormalization[T](nInputPlane,
      if (kernel == null) null else toTensor(kernel),
      threshold,
      thresval)
  }

  def createSpatialSubtractiveNormalization(nInputPlane: Int = 1,
    kernel: JTensor = null)
  : SpatialSubtractiveNormalization[T] = {
    SpatialSubtractiveNormalization[T](nInputPlane,
      if (kernel == null) null else toTensor(kernel))
  }

  def createSoftMarginCriterion(sizeAverage: Boolean = true)
  : SoftMarginCriterion[T] = {
    SoftMarginCriterion[T](sizeAverage)
  }

  def createCategoricalCrossEntropy(): CategoricalCrossEntropy[T] = {
    CategoricalCrossEntropy[T]()
  }

  //   Optimizer
  def createPoly(power: Double, maxIteration: Int): SGD.Poly = {
    SGD.Poly(power, maxIteration)
  }

  def createStep(stepSize: Int, gamma: Double): SGD.Step = {
    SGD.Step(stepSize, gamma)
  }

  def createMultiStep(stepSizes: JList[Int], gamma: Double): SGD.MultiStep = {
    SGD.MultiStep(stepSizes.asScala.toArray, gamma)
  }

  def createExponential(decayStep: Int, decayRate: Double,
    stairCase: Boolean = false): SGD.Exponential = {
    SGD.Exponential(decayStep, decayRate, stairCase)
  }

  def createDefault(): SGD.Default = {
    SGD.Default()
  }

  def createPlateau(monitor: String, factor: Float = 0.1f,
    patience: Int = 10, mode: String = "min", epsilon: Float = 1e-4f,
    cooldown: Int = 0, minLr: Float = 0): SGD.Plateau = {
    SGD.Plateau(monitor, factor, patience, mode, epsilon, cooldown, minLr)
  }

  def createWarmup(delta: Double): SGD.Warmup = {
    SGD.Warmup(delta)
  }

  def createSequentialSchedule(iterationPerEpoch: Int): SGD.SequentialSchedule = {
    SGD.SequentialSchedule(iterationPerEpoch)
  }

  def createClassNLLCriterion(weights: JTensor = null,
    sizeAverage: Boolean = true, logProbAsInput: Boolean = true)
  : ClassNLLCriterion[T] = {
    ClassNLLCriterion[T](if (weights == null) null else toTensor(weights),
      sizeAverage, logProbAsInput)
  }

  def createMSECriterion: MSECriterion[T] = {
    MSECriterion[T]()
  }

  def createAbsCriterion(sizeAverage: Boolean = true)
  : AbsCriterion[T] = {
    AbsCriterion[T](sizeAverage)
  }

  def createClassSimplexCriterion(nClasses: Int)
  : ClassSimplexCriterion[T] = {
    ClassSimplexCriterion[T](nClasses)
  }

  def createCrossEntropyCriterion(weights: JTensor = null,
    sizeAverage: Boolean = true): CrossEntropyCriterion[T] = {
    new CrossEntropyCriterion[T](if (null == weights) null else toTensor(weights), sizeAverage)
  }


  def createCosineEmbeddingCriterion(margin: Double = 0.0,
    sizeAverage: Boolean = true)
  : CosineEmbeddingCriterion[T] = {
    CosineEmbeddingCriterion[T](margin,
      sizeAverage)
  }

  def createDistKLDivCriterion(sizeAverage: Boolean = true)
  : DistKLDivCriterion[T] = {
    DistKLDivCriterion[T](sizeAverage)
  }

  def createHingeEmbeddingCriterion(margin: Double = 1,
    sizeAverage: Boolean = true)
  : HingeEmbeddingCriterion[T] = {
    HingeEmbeddingCriterion[T](margin,
      sizeAverage)
  }

  def createL1HingeEmbeddingCriterion(margin: Double = 1)
  : L1HingeEmbeddingCriterion[T] = {
    L1HingeEmbeddingCriterion[T](margin)
  }

  def createMarginCriterion(margin: Double = 1.0,
    sizeAverage: Boolean = true, squared: Boolean = false)
  : MarginCriterion[T] = {
    MarginCriterion[T](margin,
      sizeAverage, squared)
  }

  def createMarginRankingCriterion(margin: Double = 1.0,
    sizeAverage: Boolean = true)
  : MarginRankingCriterion[T] = {
    MarginRankingCriterion[T](margin,
      sizeAverage)
  }

  def createMultiCriterion()
  : MultiCriterion[T] = {
    MultiCriterion[T]()
  }

  def createMultiLabelMarginCriterion(sizeAverage: Boolean = true)
  : MultiLabelMarginCriterion[T] = {
    MultiLabelMarginCriterion[T](sizeAverage)
  }

  def createParallelCriterion(repeatTarget: Boolean = false)
  : ParallelCriterion[T] = {
    ParallelCriterion[T](repeatTarget)
  }

  def createKLDCriterion(sizeAverage: Boolean): KLDCriterion[T] = {
    KLDCriterion[T](sizeAverage)
  }

  def createGaussianCriterion(): GaussianCriterion[T] = {
    GaussianCriterion[T]()
  }

  def createSmoothL1Criterion(sizeAverage: Boolean = true)
  : SmoothL1Criterion[T] = {
    SmoothL1Criterion[T](sizeAverage)
  }

  def createSmoothL1CriterionWithWeights(sigma: Double, num: Int = 0)
  : SmoothL1CriterionWithWeights[T] = {
    SmoothL1CriterionWithWeights[T](sigma,
      num)
  }

  def createSoftmaxWithCriterion(ignoreLabel: Integer = null,
    normalizeMode: String = "VALID")
  : SoftmaxWithCriterion[T] = {
    val normM = normalizeMode match {
      case "FULL" => NormMode.FULL
      case "VALID" => NormMode.VALID
      case "BATCH_SIZE" => NormMode.BATCH_SIZE
      case "NONE" => NormMode.NONE
      case n: String =>
        throw new IllegalArgumentException(s"Only support 'FULL', " +
          s"'VALID', 'BATCH_SIZE' and 'NONE': $n")
    }
    val labelToIgnore = ignoreLabel match {
      case i: Integer => Some(i.toInt)
      case null => None
    }
    SoftmaxWithCriterion[T](labelToIgnore, normM)
  }

  def createTransformerCriterion(
           criterion: AbstractCriterion[Activity, Activity, T],
           inputTransformer: AbstractModule[Activity, Activity, T] = null,
           targetTransformer: AbstractModule[Activity, Activity, T] = null
           ): TransformerCriterion[T] = {
    TransformerCriterion(criterion, Option(inputTransformer), Option(targetTransformer))
  }

  def createDotProductCriterion(
          sizeAverage: Boolean = false): DotProductCriterion[T] = {
    DotProductCriterion[T](sizeAverage)
  }

  def createPGCriterion(
    sizeAverage: Boolean = false): PGCriterion[T] = {
    PGCriterion(sizeAverage)
  }

  def createPack(dimension: Int): Pack[T] = {
    Pack(dimension)
  }

  def createTile(dim : Int, copies : Int): Tile[T] = {
    Tile(dim, copies)
  }

  def createBinaryThreshold(th: Double, ip: Boolean): BinaryThreshold[T] = {
    BinaryThreshold(th, ip)
  }

  def setModelSeed(seed: Long): Unit = {
    RandomGenerator.RNG.setSeed(seed)
  }

  def modelEvaluate(model: AbstractModule[Activity, Activity, T],
                    valRDD: JavaRDD[Sample],
                    batchSize: Int,
                    valMethods: JList[ValidationMethod[T]])
  : JList[EvaluatedResult] = {
    val resultArray = model.evaluate(valRDD.rdd.map(toJSample(_)),
      valMethods.asScala.toArray, Some(batchSize))
    val testResultArray = resultArray.map { result =>
      EvaluatedResult(result._1.result()._1, result._1.result()._2,
        result._2.toString())
    }
    testResultArray.toList.asJava
  }


  def modelEvaluateImageFrame(model: AbstractModule[Activity, Activity, T],
                    imageFrame: ImageFrame,
                    batchSize: Int,
                    valMethods: JList[ValidationMethod[T]])
  : JList[EvaluatedResult] = {
    val resultArray = model.evaluateImage(imageFrame,
      valMethods.asScala.toArray, Some(batchSize))
    val testResultArray = resultArray.map { result =>
      EvaluatedResult(result._1.result()._1, result._1.result()._2,
        result._2.toString())
    }
    testResultArray.toList.asJava
  }

  def loadBigDL(path: String): AbstractModule[Activity, Activity, T] = {
    Module.load[T](path)
  }

  def loadBigDLModule(modulePath: String,
    weightPath : String): AbstractModule[Activity, Activity, T] = {
    Module.loadModule[T](modulePath, weightPath)
  }

  def loadTorch(path: String): AbstractModule[Activity, Activity, T] = {
    Module.loadTorch[T](path)
  }

  def loadCaffe(model: AbstractModule[Activity, Activity, T],
    defPath: String,
    modelPath: String,
    matchAll: Boolean = true): AbstractModule[Activity, Activity, T] = {
    Module.loadCaffe[T](model, defPath, modelPath, matchAll)
  }

  def loadCaffeModel(defPath: String, modelPath: String): AbstractModule[Activity, Activity, T] = {
    Module.loadCaffeModel[T](defPath, modelPath)
  }

  def loadTF(path: String, inputs: JList[String], outputs: JList[String],
        byteOrder: String, binFile: String = null,
        generatedBackward: Boolean = true): AbstractModule[Activity, Activity, T] = {
    val order = byteOrder match {
      case "little_endian" => ByteOrder.LITTLE_ENDIAN
      case "big_endian" => ByteOrder.BIG_ENDIAN
      case _ => throw new IllegalArgumentException(s"No support byte order $byteOrder")
    }
    Module.loadTF[T](path, inputs.asScala, outputs.asScala, order,
      Option(binFile), generatedBackward)
  }

  def saveTF(model: AbstractModule[Activity, Activity, T],
    inputs: JList[Any],
    path: String,
    byteOrder: String,
    dataFormat: String): Unit = {
    val order = byteOrder.toLowerCase match {
      case "little_endian" => ByteOrder.LITTLE_ENDIAN
      case "big_endian" => ByteOrder.BIG_ENDIAN
      case _ => throw new IllegalArgumentException(s"Unknown byte order $byteOrder")
    }

    val format = dataFormat.toLowerCase match {
      case "nhwc" => TensorflowDataFormat.NHWC
      case "nchw" => TensorflowDataFormat.NCHW
      case _ => throw new IllegalArgumentException(s"Unknown format $dataFormat")
    }
    val scalaInputs = inputs.asScala.map { elem =>
      val array = elem.asInstanceOf[JList[Any]]
      val name = array.get(0).asInstanceOf[String]
      val shape = array.get(1).asInstanceOf[JList[Int]]
      (name, shape.asScala)
    }
    model.saveTF(scalaInputs, path, order, format)
  }

  def predictLocal(model: AbstractModule[Activity, Activity, T],
                   features: JList[JTensor], batchSize: Int = -1): JList[JTensor] = {
    val sampleArray = toSampleArray(features.asScala.toList.map{f => toTensor(f)})
    val localPredictor = if (batchSize > 0) {
      val batchPerCore = batchSize / Engine.coreNumber()
      if (batchPerCore < 1) {
        LocalPredictor(model, batchPerCore = 1)
      } else {
        LocalPredictor(model, batchPerCore = batchPerCore)
      }
    } else {
      LocalPredictor(model)
    }
    val result = localPredictor.predict(sampleArray)
    result.map{a => toJTensor(a.asInstanceOf[Tensor[T]])}.toList.asJava
  }

  def predictLocalClass(model: AbstractModule[Activity, Activity, T],
                        features: JList[JTensor]): JList[Int] = {
    val sampleArray = toSampleArray(features.asScala.toList.map{f => toTensor(f)})
    val localPredictor = LocalPredictor(model)
    val result = localPredictor.predictClass(sampleArray)
    result.toList.asJava
  }

  def modelPredictRDD(model: AbstractModule[Activity, Activity, T],
                      dataRdd: JavaRDD[Sample], batchSize: Int = -1): JavaRDD[JTensor] = {
    val tensorRDD = model.predict(dataRdd.rdd.map(toJSample(_)), batchSize)
    val listRDD = tensorRDD.map { res =>
      val tensor = res.asInstanceOf[Tensor[T]]
      val cloneTensor = tensor.clone()
      toJTensor(cloneTensor)

    }
    new JavaRDD[JTensor](listRDD)
  }

  def modelPredictImage(model: AbstractModule[Activity, Activity, T],
    imageFrame: ImageFrame,
    featLayerName: String,
    shareBuffer: Boolean,
    batchPerPartition: Int,
    predictKey: String)
  : ImageFrame = {
    model.predictImage(imageFrame,
      featLayerName, shareBuffer, batchPerPartition, predictKey)
  }

  def evaluate(module: AbstractModule[Activity, Activity, T]):
  AbstractModule[Activity, Activity, T] = {
    module.evaluate()
  }

  def modelPredictClass(model: AbstractModule[Activity, Activity, T],
                        dataRdd: JavaRDD[Sample]): JavaRDD[Int] = {
    val sampleRdd = toJSample(dataRdd)
    val tensorRDD = model.predictClass(sampleRdd)
    new JavaRDD[Int](tensorRDD)
  }

  def modelForward(model: AbstractModule[Activity, Activity, T],
    input: JList[_ <: Object],
    inputIsTable: Boolean): JList[JTensor] = {
    val inputActivity = jTensorsToActivity(input, inputIsTable)
    val outputActivity = model.forward(inputActivity)
    activityToJTensors(outputActivity)
  }

  def modelBackward(model: AbstractModule[Activity, Activity, T],
    input: JList[_ <: Object],
    inputIsTable: Boolean,
    gradOutput: JList[_ <: Object],
    gradOutputIsTable: Boolean): JList[JTensor] = {
    val inputActivity = jTensorsToActivity(input, inputIsTable)
    val gradOutputActivity = jTensorsToActivity(gradOutput, gradOutputIsTable)
    val outputActivity = model.backward(inputActivity, gradOutputActivity)
    activityToJTensors(outputActivity)
  }


  def modelSave(module: AbstractModule[Activity, Activity, T],
    path: String, overWrite: Boolean): Unit = {
    module.save(path, overWrite)
  }

  def saveBigDLModule(module: AbstractModule[Activity, Activity, T],
    modulePath: String, weightPath: String, overWrite: Boolean): Unit = {
    module.saveModule(modulePath, weightPath, overWrite)
  }

  def saveCaffe(module: AbstractModule[Activity, Activity, T],
    prototxtPath: String, modelPath: String,
    useV2: Boolean = true, overwrite: Boolean = false): Unit = {
    module.saveCaffe(prototxtPath, modelPath, useV2, overwrite)
  }

  def criterionForward(criterion: AbstractCriterion[Activity, Activity, T],
    input: JList[_ <: Object],
    inputIsTable: Boolean,
    target: JList[_ <: Object],
    targetIsTable: Boolean): T = {
    val inputActivity = jTensorsToActivity(input, inputIsTable)
    val targetActivity = jTensorsToActivity(target, targetIsTable)
    return criterion.forward(inputActivity, targetActivity)
  }

  def criterionBackward(criterion: AbstractCriterion[Activity, Activity, T],
    input: JList[_ <: Object],
    inputIsTable: Boolean,
    target: JList[_ <: Object],
    targetIsTable: Boolean): JList[JTensor] = {
    val inputActivity = jTensorsToActivity(input, inputIsTable)
    val targetActivity = jTensorsToActivity(target, targetIsTable)
    val outputActivity = criterion.backward(inputActivity, targetActivity)
    activityToJTensors(outputActivity)
  }

  def modelGetParameters(model: AbstractModule[Activity, Activity, T])
  : JMap[Any, JMap[Any, JList[JList[Any]]]] = {
    model.getParametersTable().getState().mapValues {
      case name2Values: Table =>
        name2Values.getState().mapValues {
          case t: Tensor[T] =>
            val tensorClone = t.clone()
            val item = List(tensorClone.storage().toList.asJava.asInstanceOf[JList[Any]],
              tensorClone.size().toList.asJava.asInstanceOf[JList[Any]]).asJava
            item
        }.asJava
    }.asJava
  }

  def createMaxEpoch(max: Int): Trigger = {
    Trigger.maxEpoch(max)
  }

  def createEveryEpoch(): Trigger = {
    Trigger.everyEpoch
  }

  def createSeveralIteration(interval: Int): Trigger = {
    Trigger.severalIteration(interval)
  }

  def createMaxIteration(max: Int): Trigger = {
    Trigger.maxIteration(max)
  }

  def createMaxScore(max: Float): Trigger = {
    Trigger.maxScore(max)
  }

  def createMinLoss(min: Float): Trigger = {
    Trigger.minLoss(min)
  }

  def createTriggerAnd(first: Trigger, others: JList[Trigger]): Trigger = {
    Trigger.and(first, others.asScala: _*)
  }

  def createTriggerOr(first: Trigger, others: JList[Trigger]): Trigger = {
    Trigger.or(first, others.asScala: _*)
  }

  def createTop1Accuracy(): ValidationMethod[T] = {
    new Top1Accuracy()
  }

  def createHitRatio(k: Int = 10, negNum: Int = 100): ValidationMethod[T] = {
    new HitRatio(k, negNum)
  }

  def createNDCG(k: Int = 10, negNum: Int = 100): ValidationMethod[T] = {
    new NDCG(k, negNum)
  }

  def createTreeNNAccuracy(): ValidationMethod[T] = {
    new TreeNNAccuracy()
  }

  def createTop5Accuracy(): ValidationMethod[T] = {
    new Top5Accuracy()
  }

  def createMeanAveragePrecision(k: Int, classes: Int): ValidationMethod[T] = {
    new MeanAveragePrecision(k, classes)
  }

  def createMeanAveragePrecisionObjectDetection(classes: Int, iou: Float, useVoc2007: Boolean,
    skipClass: Int): ValidationMethod[T] = {
    new MeanAveragePrecisionObjectDetection(classes, iouThres = Array(iou),
      theType = if (useVoc2007) MAPPascalVoc2007 else MAPPascalVoc2010, skipClass = skipClass)
  }

  def createLoss(criterion: Criterion[T]): ValidationMethod[T] = {
    new Loss(criterion)
  }

  def createMAE(): ValidationMethod[T] = {
    new MAE()
  }

  def createSGD(learningRate: Double = 1e-3,
    learningRateDecay: Double = 0.0,
    weightDecay: Double = 0.0,
    momentum: Double = 0.0,
    dampening: Double = Double.MaxValue,
    nesterov: Boolean = false,
    leaningRateSchedule: SGD.LearningRateSchedule = SGD.Default(),
    learningRates: JTensor = null,
    weightDecays: JTensor = null): SGD[T] = {
    val p1 = if (learningRates == null) null else toTensor(learningRates)
    val p2 = if (weightDecays == null) null else toTensor(weightDecays)
    new SGD[T](learningRate, learningRateDecay, weightDecay, momentum, dampening,
      nesterov, leaningRateSchedule, p1, p2)
  }

  def createAdagrad(learningRate: Double = 1e-3,
    learningRateDecay: Double = 0.0,
    weightDecay: Double = 0.0): Adagrad[T] = {
    new Adagrad[T](learningRate, learningRateDecay, weightDecay)
  }

  def createLBFGS(maxIter: Int = 20,
    maxEval: Double = Double.MaxValue,
    tolFun: Double = 1e-5,
    tolX: Double = 1e-9,
    nCorrection: Int = 100,
    learningRate: Double = 1.0,
    verbose: Boolean = false,
    lineSearch: LineSearch[T] = null,
    lineSearchOptions: JMap[Any, Any] = null): LBFGS[T] = {
    val p1 = if (lineSearch == null) None else Option(lineSearch)
    val p2 = if (lineSearchOptions == null) None else Option(T(lineSearchOptions))
    new LBFGS[T](maxIter, maxEval, tolFun, tolX, nCorrection, learningRate, verbose, p1, p2)
  }

  def createAdadelta(decayRate: Double = 0.9, Epsilon: Double = 1e-10): Adadelta[T] = {
    new Adadelta[T](decayRate, Epsilon)
  }

  def createAdam(
    learningRate: Double = 1e-3,
    learningRateDecay: Double = 0.0,
    beta1: Double = 0.9,
    beta2: Double = 0.999,
    Epsilon: Double = 1e-8): Adam[T] = {
    new Adam[T](learningRate, learningRateDecay, beta1, beta2, Epsilon)
  }

  def createParallelAdam(
        learningRate: Double = 1e-3,
        learningRateDecay: Double = 0.0,
        beta1: Double = 0.9,
        beta2: Double = 0.999,
        Epsilon: Double = 1e-8,
        parallelNum: Int = Engine.coreNumber()): ParallelAdam[T] = {
    new ParallelAdam[T](learningRate, learningRateDecay, beta1, beta2, Epsilon, parallelNum)
  }

  def createFtrl(
      learningRate: Double = 1e-3,
      learningRatePower: Double = -0.5,
      initialAccumulatorValue: Double = 0.1,
      l1RegularizationStrength: Double = 0.0,
      l2RegularizationStrength: Double = 0.0,
      l2ShrinkageRegularizationStrength: Double = 0.0): Ftrl[T] = {
    new Ftrl[T](learningRate,
      learningRatePower,
      initialAccumulatorValue,
      l1RegularizationStrength,
      l2RegularizationStrength,
      l2ShrinkageRegularizationStrength)
  }

  def createAdamax(
    learningRate: Double = 0.002,
    beta1: Double = 0.9,
    beta2: Double = 0.999,
    Epsilon: Double = 1e-38): Adamax[T] = {
    new Adamax(learningRate, beta1, beta2, Epsilon)
  }

  def createRMSprop(
    learningRate: Double = 1e-2,
    learningRateDecay: Double = 0.0,
    decayRate: Double = 0.99,
    Epsilon: Double = 1e-8): RMSprop[T] = {
    new RMSprop[T](learningRate, learningRateDecay, decayRate, Epsilon)
  }

  def loadOptimMethod(path: String): OptimMethod[T] = {
    OptimMethod.load[T](path)
  }

  def saveOptimMethod(method: OptimMethod[T], path: String,
    overWrite: Boolean = false): Unit = {
    method.save(path, overWrite)
  }

  /**
   * Save tensor dictionary to a Java hashmap object file
   */
  def saveTensorDictionary(tensors: JHashMap[String, JTensor], path: String): Unit = {
    File.save(tensors, path, true)
  }

  def trainTF(
               modelPath: String,
               output: String,
               samples: JavaRDD[Sample],
               optMethod: OptimMethod[T],
               criterion: Criterion[T],
               batchSize: Int,
               endWhen: Trigger): AbstractModule[Activity, Activity, T] = {
    val nodeList = parse(modelPath)

    val context = new Context[T]()
    val session = new BigDLSessionImpl[T](nodeList.asScala, context, ByteOrder.LITTLE_ENDIAN)
    val dataset = batching(DataSet.rdd(toJSample(samples)),
      batchSize).asInstanceOf[DistributedDataSet[MiniBatch[T]]]
    val model = session.train(Seq(output), dataset,
      optMethod, criterion, endWhen)
    model
  }

  def createLocalOptimizer(features: JList[JTensor],
                           y: JTensor,
                           model: AbstractModule[Activity, Activity, T],
                           criterion: Criterion[T],
                           optimMethod: JMap[String, OptimMethod[T]],
                           endTrigger: Trigger,
                           batchSize: Int,
                           localCores: Int): Optimizer[T, MiniBatch[T]] = {
    val sampleArray = toSampleArray(features.asScala.toList.map{f => toTensor(f)}, toTensor(y))
    val optimizer = new LocalOptimizer[T](
      model,
      batching(DataSet.array(sampleArray), batchSize)
        .asInstanceOf[LocalDataSet[MiniBatch[T]]],
      criterion
    ).asInstanceOf[Optimizer[T, MiniBatch[T]]]
    Engine.setNodeAndCore(1, localCores)
    enrichOptimizer[T](optimizer, endTrigger, optimMethod.asScala.toMap)
  }

  def createDistriOptimizer(model: AbstractModule[Activity, Activity, T],
                            trainingRdd: JavaRDD[Sample],
                            criterion: Criterion[T],
                            optimMethod: JMap[String, OptimMethod[T]],
                            endTrigger: Trigger,
                            batchSize: Int): Optimizer[T, MiniBatch[T]] = {
    val sampleRDD = toJSample(trainingRdd)
    val optimizer = Optimizer(
      model = model,
      dataset = batching(DataSet.rdd(sampleRDD), batchSize)
        .asInstanceOf[DistributedDataSet[MiniBatch[T]]],
      criterion = criterion
    ).asInstanceOf[Optimizer[T, MiniBatch[T]]]
    enrichOptimizer(optimizer, endTrigger, optimMethod.asScala.toMap)
  }

  def createDistriOptimizerFromDataSet(model: AbstractModule[Activity, Activity, T],
    trainDataSet: DataSet[ImageFeature],
    criterion: Criterion[T],
    optimMethod: JMap[String, OptimMethod[T]],
    endTrigger: Trigger,
    batchSize: Int): Optimizer[T, MiniBatch[T]] = {
    val dataSet = trainDataSet -> ImageFeatureToMiniBatch[T](batchSize)
    val optimizer = Optimizer(
      model = model,
      dataset = dataSet.asInstanceOf[DistributedDataSet[MiniBatch[T]]],
      criterion = criterion
    ).asInstanceOf[Optimizer[T, MiniBatch[T]]]
    enrichOptimizer(optimizer, endTrigger, optimMethod.asScala.toMap)
  }

  def featureTransformDataset(dataset: DataSet[ImageFeature],
    transformer: FeatureTransformer): DataSet[ImageFeature] = {
    dataset -> transformer
  }

  def createL1L2Regularizer(l1: Double, l2: Double): L1L2Regularizer[T] = {
    L1L2Regularizer[T](l1, l2)
  }

  def createL1Regularizer(l1: Double): L1Regularizer[T] = {
    L1Regularizer[T](l1)
  }

  def createL2Regularizer(l2: Double): L2Regularizer[T] = {
    L2Regularizer[T](l2)
  }

  def setValidation(optimizer: Optimizer[T, MiniBatch[T]],
                    batchSize: Int,
                    trigger: Trigger,
                    valRdd: JavaRDD[Sample],
                    vMethods: JList[ValidationMethod[T]]): Unit = {
    val sampleRDD = toJSample(valRdd)
    optimizer.setValidation(trigger, batching(DataSet.rdd(sampleRDD), batchSize.toInt),
      vMethods.asScala.toArray)
  }

  def setValidationFromDataSet(optimizer: Optimizer[T, MiniBatch[T]],
    batchSize: Int,
    trigger: Trigger,
    valDataSet: DataSet[ImageFeature],
    vMethods: JList[ValidationMethod[T]]): Unit = {
    val dataSet = valDataSet -> ImageFeatureToMiniBatch[T](batchSize)
    optimizer.setValidation(trigger, dataSet,
      vMethods.asScala.toArray)
  }

  def setValidation(optimizer: Optimizer[T, MiniBatch[T]],
                    batchSize: Int,
                    trigger: Trigger,
                    xVal: JList[JTensor],
                    yVal: JTensor,
                    vMethods: JList[ValidationMethod[T]]): Unit = {

    val sampleArray = toSampleArray(xVal.asScala.toList.map{f => toTensor(f)}, toTensor(yVal))
    optimizer.setValidation(trigger, batching(DataSet.array(sampleArray), batchSize),
      vMethods.asScala.toArray)
  }

  def setTrainData(optimizer: Optimizer[T, MiniBatch[T]],
                 trainingRdd: JavaRDD[Sample],
                 batchSize: Int): Unit = {
    val sampleRDD = toJSample(trainingRdd)
    optimizer.setTrainData(sampleRDD, batchSize)
  }

  def setCriterion(optimizer: Optimizer[T, MiniBatch[T]],
                   criterion: Criterion[T]): Unit = {
    optimizer.setCriterion(criterion)
  }

  def setCheckPoint(optimizer: Optimizer[T, MiniBatch[T]],
    trigger: Trigger,
    checkPointPath: String,
    isOverwrite: Boolean): Unit = {
    optimizer.setCheckpoint(checkPointPath, trigger)
    if (isOverwrite) {
      optimizer.overWriteCheckpoint()
    }
  }

  def setTrainSummary(optimizer: Optimizer[T, MiniBatch[T]], summary: TrainSummary): Unit = {
    optimizer.setTrainSummary(summary)
  }

  def setValSummary(optimizer: Optimizer[T, MiniBatch[T]], summary: ValidationSummary): Unit = {
    optimizer.setValidationSummary(summary)
  }

  def summaryReadScalar(summary: Summary, tag: String): JList[JList[Any]] = {
    val result = summary.readScalar(tag)
    result.toList.map { item =>
      List(item._1, item._2, item._3).asJava.asInstanceOf[JList[Any]]
    }.asJava
  }

  def summarySetTrigger(
    summary: TrainSummary,
    summaryName: String,
    trigger: Trigger): TrainSummary = {
    summary.setSummaryTrigger(summaryName, trigger)
    summary
  }

  def createTrainSummary(logDir: String,
    appName: String): TrainSummary = {
    new TrainSummary(logDir, appName)
  }

  def createValidationSummary(logDir: String,
    appName: String): ValidationSummary = {
    new ValidationSummary(logDir, appName)
  }

  def createModel(input: JList[ModuleNode[T]],
                  output: JList[ModuleNode[T]]): Graph[T] = {
    Graph(input.asScala.toArray, output.asScala.toArray)
  }

  def createModelPreprocessor(preprocessor: AbstractModule[Activity, Activity, T],
    trainable: AbstractModule[Activity, Activity, T]): Graph[T] = {
    Graph(preprocessor, trainable)
  }

  def createNode(module: AbstractModule[Activity, Activity, T],
    x: JList[ModuleNode[T]]): ModuleNode[T] = {
    if (null == x || x.isEmpty) {
      module.inputs()
    } else {
      module.inputs(x.asScala: _*)
    }
  }

  def createInput(): ModuleNode[T] = {
    Input()
  }

  def initEngine(): Unit = {
    Engine.init
  }

  def getEngineType(): String = {
    Engine.getEngineType().toString
  }

  def getNodeAndCoreNumber(): Array[Int] = {
    Array(Engine.nodeNumber(), Engine.coreNumber())
  }

  def setOptimizerVersion(version: String): Unit = {
    version.toLowerCase() match {
      case "optimizerv1" => Engine.setOptimizerVersion(OptimizerV1)
      case "optimizerv2" => Engine.setOptimizerVersion(OptimizerV2)
    }
  }

  def getOptimizerVersion(): String = {
    Engine.getOptimizerVersion().toString
  }

  def setWeights(model: AbstractModule[Activity, Activity, T], weights: JList[JTensor]): Unit = {
    val weightTensor = weights.asScala.toArray.map(toTensor(_))
    model.setWeightsBias(weightTensor)
  }

  def getWeights(model: AbstractModule[Activity, Activity, T]): JList[JTensor] = {
    val weights = model.getWeightsBias()
    if (weights != null) {
      weights.map(toJTensor(_)).toList.asJava
    } else {
      null
    }
  }

  def updateParameters(model: AbstractModule[Activity, Activity, T], lr: Double): Unit = {
    val (w, g) = model.getParameters()
    w.add(ev.negative(ev.fromType(lr)), g)
  }

  def uniform(a: Double, b: Double, size: JList[Int]): JTensor = {
    val result = Tensor[T]().resize(size.asScala.toArray)
    result.apply1(i => ev.fromType(RandomGenerator.RNG.uniform(a, b)))
    toJTensor(result)
  }

  def createZeros(): Zeros.type = {
    Zeros
  }

  def createOnes(): Ones.type = {
    Ones
  }

  def createConstInitMethod(value: Double): ConstInitMethod = {
    ConstInitMethod(value)
  }

  def createRandomUniform(lower: Double, upper: Double): InitializationMethod = {
    RandomUniform(lower, upper)
  }

  def createRandomUniform(): InitializationMethod = {
    RandomUniform
  }

  def createRandomNormal(mean: Double, stdv: Double): RandomNormal = {
    RandomNormal(mean, stdv)
  }

  def createXavier(): Xavier.type = {
    Xavier
  }

  def createMsraFiller(varianceNormAverage: Boolean = true): MsraFiller = {
    MsraFiller(varianceNormAverage)
  }

  def createBilinearFiller(): BilinearFiller.type = {
    BilinearFiller
  }

  def createHardSigmoid : HardSigmoid[T] = {
    HardSigmoid()
  }

  def createMeanAbsolutePercentageCriterion: MeanAbsolutePercentageCriterion[T] = {
    MeanAbsolutePercentageCriterion()
  }

  def createMeanSquaredLogarithmicCriterion: MeanSquaredLogarithmicCriterion[T] = {
    MeanSquaredLogarithmicCriterion()
  }

  def createKullbackLeiblerDivergenceCriterion: KullbackLeiblerDivergenceCriterion[T] = {
    KullbackLeiblerDivergenceCriterion()
  }

  def createPoissonCriterion: PoissonCriterion[T] = {
    PoissonCriterion()
  }

  def setInitMethod(layer: Initializable, weightInitMethod: InitializationMethod,
    biasInitMethod: InitializationMethod): layer.type = {
    layer.setInitMethod(weightInitMethod, biasInitMethod)
  }

  def setInitMethod(layer: Initializable,
    initMethods: JArrayList[InitializationMethod]): layer.type = {
    layer.setInitMethod(initMethods.asScala.toArray)
  }

  def getHiddenState(rec: Recurrent[T]): JActivity = {
    JActivity(rec.getHiddenState())
  }

  def freeze(model: AbstractModule[Activity, Activity, T], freezeLayers: JList[String])
  : AbstractModule[Activity, Activity, T] = {
    if (null == freezeLayers) model.freeze() else model.freeze(freezeLayers.asScala: _*)
  }

  def unFreeze(model: AbstractModule[Activity, Activity, T],
    names: JList[String]): AbstractModule[Activity, Activity, T] = {
    if (names == null) {
      model.unFreeze()
    } else {
      model.unFreeze(names.asScala: _*)
    }
  }

  def setStopGradient(model: Graph[T], layers: JList[String]): Graph[T] = {
    model.stopGradient(layers.asScala.toArray)
  }

  def saveGraphTopology(model: Graph[T], logPath: String): Graph[T] = {
    model.saveGraphTopology(logPath)
  }

  def setInputFormats(graph: StaticGraph[T], inputFormat: JList[Int]): StaticGraph[T] = {
    graph.setInputFormats(inputFormat.asScala.toList)
  }

  def setOutputFormats(graph: StaticGraph[T], outputFormat: JList[Int]): StaticGraph[T] = {
    graph.setOutputFormats(outputFormat.asScala.toList)
  }

  def createResizeBilinear(
    outputHeight: Int,
    outputWidth: Int,
    alignCorner: Boolean,
    dataFormat: String
  ): ResizeBilinear[T] = {
    ResizeBilinear[T](outputHeight,
      outputWidth,
      alignCorner, DataFormat.apply(dataFormat))
  }

  def createMultiRNNCell(cells: JList[Cell[T]]): MultiRNNCell[T] = {
    MultiRNNCell(cells.asScala.toArray)
  }

  def createHighway(size: Int, withBias: Boolean,
    activation: TensorModule[T] = null,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null): Graph[T] = {
    Highway(size, withBias, activation, wRegularizer, bRegularizer)
  }

  def createUpSampling3D(size: JList[Int]): UpSampling3D[T] = {
    UpSampling3D(size.asScala.toArray)
  }

  def createCropping2D(
      heightCrop: JList[Int],
      widthCrop: JList[Int],
      dataFormat: String = "NCHW"): Cropping2D[T] = {
    Cropping2D(heightCrop.asScala.toArray, widthCrop.asScala.toArray, DataFormat(dataFormat))
  }

  def createCropping3D(
      dim1Crop: JList[Int],
      dim2Crop: JList[Int],
      dim3Crop: JList[Int],
      dataFormat: String = Cropping3D.CHANNEL_FIRST): Cropping3D[T] = {
    Cropping3D(
      dim1Crop.asScala.toArray, dim2Crop.asScala.toArray, dim3Crop.asScala.toArray, dataFormat)
  }

  def redirectSparkLogs(logPath: String): Unit = {
    LoggerFilter.redirectSparkInfoLogs(logPath)
  }

  def showBigDlInfoLogs(): Unit = {
    Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)
  }

  def quantize(module: AbstractModule[Activity, Activity, T]): Module[T] = {
    module.quantize()
  }

  def createDLEstimator(model: Module[T], criterion: Criterion[T],
                        featureSize: JArrayList[Int],
                        labelSize: JArrayList[Int]): DLEstimator[T] = {
    new DLEstimator[T](model, criterion, featureSize.asScala.toArray, labelSize.asScala.toArray)
  }

  def createDLClassifier(model: Module[T], criterion: Criterion[T],
                         featureSize: JArrayList[Int],
                         labelSize: JArrayList[Int]): DLClassifier[T] = {
    new DLClassifier[T](model, criterion, featureSize.asScala.toArray)
  }

  def fitEstimator(estimator: DLEstimator[T], dataSet: DataFrame): DLModel[T] = {
    estimator.fit(dataSet)
  }

  def fitClassifier(classifier: DLClassifier[T], dataSet: DataFrame): DLModel[T] = {
    classifier.fit(dataSet)
  }

  def setBatchSizeDLEstimator(estimator: DLEstimator[T], batchSize: Int): DLEstimator[T] = {
    estimator.setBatchSize(batchSize)
  }

  def setBatchSizeDLClassifier(classifier: DLClassifier[T], batchSize: Int): DLClassifier[T] = {
    classifier.setBatchSize(batchSize)
  }

  def setMaxEpochDLEstimator(estimator: DLEstimator[T], maxEpoch: Int): DLEstimator[T] = {
    estimator.setMaxEpoch(maxEpoch)
  }

  def setMaxEpochDLClassifier(classifier: DLClassifier[T], maxEpoch: Int): DLClassifier[T] = {
    classifier.setMaxEpoch(maxEpoch)
  }

  def setLearningRateDLEstimator(estimator: DLEstimator[T], lr: Double): DLEstimator[T] = {
    estimator.setLearningRate(lr)
  }

  def setLearningRateDLClassifier(classifier: DLClassifier[T], lr: Double): DLClassifier[T] = {
    classifier.setLearningRate(lr)
  }

  def createDLModel(model: Module[T], featureSize: JArrayList[Int]): DLModel[T] = {
    new DLModel[T](model, featureSize.asScala.toArray)
  }

  def createDLClassifierModel(model: Module[T],
                              featureSize: JArrayList[Int]): DLClassifierModel[T] = {
    new DLClassifierModel[T](model, featureSize.asScala.toArray)
  }

  def dlModelTransform(dlModel: DLModel[T], dataSet: DataFrame): DataFrame = {
    dlModel.transform(dataSet)
  }

  def dlClassifierModelTransform(dlClassifierModel: DLClassifierModel[T],
                                 dataSet: DataFrame): DataFrame = {
    dlClassifierModel.transform(dataSet)
  }

  def setFeatureSizeDLModel(dlModel: DLModel[T], featureSize: JArrayList[Int]): DLModel[T] = {
    dlModel.setFeatureSize(featureSize.asScala.toArray)
  }

  def setFeatureSizeDLClassifierModel(dlClassifierModel: DLClassifierModel[T],
                                      featureSize: JArrayList[Int]): DLClassifierModel[T] = {
    dlClassifierModel.setFeatureSize(featureSize.asScala.toArray)
  }

  def setBatchSizeDLModel(dlModel: DLModel[T], batchSize: Int): DLModel[T] = {
    dlModel.setBatchSize(batchSize)
  }

  def setBatchSizeDLClassifierModel(dlClassifierModel: DLClassifierModel[T],
                                    batchSize: Int): DLClassifierModel[T] = {
    dlClassifierModel.setBatchSize(batchSize)
  }

  def findGraphNode(model: Graph[T], name: String): ModuleNode[T] = {
    model.node(name)
  }

  def getContainerModules(module: Container[Activity, Activity, T])
  : JList[AbstractModule[Activity, Activity, T]] = {
    module match {
      case m: KerasModel[T] =>
        m.getSubModules().asJava
      case kl: KerasLayer[Activity, Activity, T] =>
        throw new RuntimeException(s"There's no sub modules for ${kl}")
      case _ =>
        module.modules.toList.asJava
    }
  }

  def getFlattenModules(module: Container[Activity, Activity, T],
  includeContainer: Boolean)
  : JList[AbstractModule[Activity, Activity, T]] = {
    val result = ArrayBuffer[AbstractModule[Activity, Activity, T]]()
    doGetFlattenModules(module, includeContainer, result)
    result.toList.asJava
  }

  // TODO: refactor Container and KerasLayer to simplify this logic
  private def hasSubModules(module: AbstractModule[Activity, Activity, T]) = {
    module match {
      case km: KerasModel[T] => true
      case kl: KerasLayer[Activity, Activity, T] => false
      case c: Container[_, _, _] => true
      case _ => false
    }
  }

  private def doGetFlattenModules(module: Container[Activity, Activity, T],
    includeContainer: Boolean,
    result: ArrayBuffer[AbstractModule[Activity, Activity, T]]): Unit = {
    getContainerModules(module).asScala.foreach {m =>
      if (hasSubModules(m)) {
        doGetFlattenModules(m.asInstanceOf[Container[Activity, Activity, T]],
          includeContainer,
          result)
      } else {
        result.append(m)
      }
    }
    if (includeContainer) {
      result.append(module)
    }
  }

  def isWithWeights(module: Module[T]): Boolean = {
    val weights = module.getWeightsBias()
    return weights != null && !weights.isEmpty
  }

  def setRunningMean(module: BatchNormalization[T], runningMean: JTensor): Unit = {
    module.runningMean.set(toTensor(runningMean))
  }

  def setRunningStd(module: BatchNormalization[T], runningStd: JTensor): Unit = {
    module.runningVar.set(toTensor(runningStd))
  }

  def getRunningMean(module: BatchNormalization[T]): JTensor = {
    toJTensor(module.runningMean)
  }

  def getRunningStd(module: BatchNormalization[T]): JTensor = {
    toJTensor(module.runningVar)
  }

  def createMasking(maskValue: Double)
  : Masking[T] = {
    Masking[T](maskValue)
  }

  def createMaxout(inputSize: Int, outputSize: Int, maxoutNumber: Int, withBias: Boolean = true,
    wRegularizer: Regularizer[T] = null, bRegularizer: Regularizer[T] = null,
    initWeight: Tensor[T] = null, initBias: Tensor[T] = null)
  : Maxout[T] = {
    Maxout[T](inputSize, outputSize, maxoutNumber, withBias, wRegularizer, bRegularizer,
      initWeight, initBias)
  }

  def createCosineProximityCriterion(): CosineProximityCriterion[T] = {
    CosineProximityCriterion[T]()
  }

  def createPriorBox(minSizes: JList[Double], maxSizes: JList[Double] = null,
    aspectRatios: JList[Double] = null, isFlip: Boolean = true, isClip: Boolean = false,
    variances: JList[Double] = null, offset: Float = 0.5f,
    imgH: Int = 0, imgW: Int = 0, imgSize: Int = 0,
    stepH: Float = 0, stepW: Float = 0, step: Float = 0): PriorBox[T] = {
    val maxS = if (maxSizes == null) null else maxSizes.asScala.toArray.map(_.toFloat)
    val aspectR = if (aspectRatios == null) null else aspectRatios.asScala.toArray.map(_.toFloat)
    val vars = if (variances == null) null else variances.asScala.toArray.map(_.toFloat)
    new PriorBox[T](minSizes.asScala.toArray.map(_.toFloat),
      maxS, aspectR, isFlip, isClip, vars, offset, imgH, imgW, imgSize, stepH, stepW, step)
  }

  def createNormalizeScale(p: Double, eps: Double = 1e-10, scale: Double, size: JList[Int],
    wRegularizer: Regularizer[T] = null): NormalizeScale[T] =
    new NormalizeScale[T](p, eps, scale, size.asScala.toArray, wRegularizer)

  def createDetectionOutputSSD(nClasses: Int,
    shareLocation: Boolean,
    bgLabel: Int,
    nmsThresh: Double,
    nmsTopk: Int,
    keepTopK: Int,
    confThresh: Double,
    varianceEncodedInTarget: Boolean,
    confPostProcess: Boolean): DetectionOutputSSD[T] =
    new DetectionOutputSSD[T](nClasses, shareLocation, bgLabel, nmsThresh.toFloat,
      nmsTopk, keepTopK, confThresh.toFloat, varianceEncodedInTarget, confPostProcess)

  def createDetectionOutputFrcnn(nmsThresh: Float = 0.3f, nClasses: Int,
    bboxVote: Boolean, maxPerImage: Int = 100, thresh: Double = 0.05): DetectionOutputFrcnn = {
    new DetectionOutputFrcnn(nmsThresh, nClasses, bboxVote, maxPerImage, thresh)
  }

  def createProposal(preNmsTopN: Int, postNmsTopN: Int,
    ratios: JList[Double], scales: JList[Double],
    rpnPreNmsTopNTrain: Int = 12000, rpnPostNmsTopNTrain: Int = 2000): Proposal = {
    new Proposal(preNmsTopN, postNmsTopN, ratios.asScala.toArray.map(_.toFloat),
      scales.asScala.toArray.map(_.toFloat), rpnPreNmsTopNTrain, rpnPostNmsTopNTrain)
  }

  def createHFlip(): HFlip = {
    HFlip()
  }

  def createResize(resizeH: Int, resizeW: Int, resizeMode: Int = Imgproc.INTER_LINEAR,
    useScaleFactor: Boolean): Resize = {
    Resize(resizeH, resizeW, resizeMode, useScaleFactor)
  }

  def createColorJitter(brightnessProb: Double = 0.5, brightnessDelta: Double = 32,
    contrastProb: Double = 0.5, contrastLower: Double = 0.5, contrastUpper: Double = 1.5,
    hueProb: Double = 0.5, hueDelta: Double = 18,
    saturationProb: Double = 0.5, saturationLower: Double = 0.5, saturationUpper: Double = 1.5,
    randomOrderProb: Double = 0, shuffle: Boolean = false): ColorJitter = {
    ColorJitter(brightnessProb, brightnessDelta, contrastProb,
      contrastLower, contrastUpper, hueProb, hueDelta, saturationProb,
      saturationLower, saturationUpper, randomOrderProb, shuffle)
  }

  def createBrightness(deltaLow: Double, deltaHigh: Double): Brightness = {
    Brightness(deltaLow, deltaHigh)
  }

  def createChannelOrder(): ChannelOrder = {
    ChannelOrder()
  }

  def createContrast(deltaLow: Double, deltaHigh: Double): Contrast = {
    Contrast(deltaLow, deltaHigh)
  }

  def createRandomCrop(cropWidth: Int, cropHeight: Int, isClip: Boolean): RandomCrop = {
    RandomCrop(cropWidth, cropHeight, isClip)
  }

  def createCenterCrop(cropWidth: Int, cropHeight: Int, isClip: Boolean): CenterCrop = {
    CenterCrop(cropWidth, cropHeight, isClip)
  }

  def createFixedCrop(wStart: Double,
    hStart: Double, wEnd: Double, hEnd: Double, normalized: Boolean,
    isClip: Boolean): FixedCrop = {
    FixedCrop(wStart.toFloat, hStart.toFloat, wEnd.toFloat, hEnd.toFloat, normalized, isClip)
  }

  def createDetectionCrop(roiKey: String, normalized: Boolean): DetectionCrop = {
    DetectionCrop(roiKey, normalized)
  }

  def createExpand(meansR: Int = 123, meansG: Int = 117, meansB: Int = 104,
    minExpandRatio: Double = 1.0,
    maxExpandRatio: Double = 4.0): Expand = {
    Expand(meansR, meansG, meansB, minExpandRatio, maxExpandRatio)
  }

  def createRandomAspectScale(scales: JList[Int], scaleMultipleOf: Int = 1,
    maxSize: Int = 1000): RandomAspectScale = {
    RandomAspectScale(scales.asScala.toArray, scaleMultipleOf, maxSize)
  }

  def createHue(deltaLow: Double, deltaHigh: Double): Hue = {
    Hue(deltaLow, deltaHigh)
  }

  def createRandomTransformer(transformer: FeatureTransformer, prob: Double): RandomTransformer = {
    RandomTransformer(transformer, prob)
  }

  def createSaturation(deltaLow: Double, deltaHigh: Double): Saturation = {
    Saturation(deltaLow, deltaHigh)
  }

  def createRandomSampler(): FeatureTransformer = {
    RandomSampler()
  }

  def createChannelNormalize(meanR: Double, meanG: Double, meanB: Double,
    stdR: Double = 1, stdG: Double = 1, stdB: Double = 1): FeatureTransformer = {
    ChannelNormalize(meanR.toFloat, meanG.toFloat, meanB.toFloat,
      stdR.toFloat, stdG.toFloat, stdB.toFloat)
  }

  def createAspectScale(scale: Int,
    scaleMultipleOf: Int,
    maxSize: Int,
    resizeMode: Int = 1,
    useScaleFactor: Boolean = true,
    minScale: Double = -1): FeatureTransformer = {
    val minS = if (minScale == -1) None else Some(minScale.toFloat)
    AspectScale(scale, scaleMultipleOf, maxSize, resizeMode, useScaleFactor, minS)
  }

  def createFiller(startX: Double, startY: Double, endX: Double, endY: Double,
    value: Int = 255): Filler = {
    Filler(startX.toFloat, startY.toFloat, endX.toFloat, endY.toFloat, value)
  }

  def createPixelNormalize(means: JList[Double]): PixelNormalizer = {
    PixelNormalizer(means.asScala.toArray.map(_.toFloat))
  }

  def createRoiProject(needMeetCenterConstraint: Boolean): RoiProject = {
    RoiProject(needMeetCenterConstraint)
  }

  def createRoiResize(normalized: Boolean): RoiResize = {
    RoiResize(normalized)
  }

  def createRoiHFlip(normalized: Boolean = true): RoiHFlip = {
    RoiHFlip(normalized)
  }

  def createRoiNormalize(): RoiNormalize = {
    RoiNormalize()
  }

  def createFixExpand(eh: Int, ew: Int): FixExpand = {
    FixExpand(eh, ew)
  }

  def createChannelScaledNormalizer(meanR: Int, meanG: Int, meanB: Int, scale: Double)
    : ChannelScaledNormalizer = {
    ChannelScaledNormalizer(meanR, meanG, meanB, scale)
  }

  def createRandomAlterAspect(min_area_ratio: Float,
                              max_area_ratio: Int,
                              min_aspect_ratio_change: Float,
                              interp_mode: String,
                              cropLength: Int)
  : RandomAlterAspect = {
    RandomAlterAspect(min_area_ratio, max_area_ratio, min_aspect_ratio_change,
      interp_mode, cropLength)
  }

  def createRandomCropper(cropWidth: Int, cropHeight: Int,
                          mirror: Boolean, cropperMethod: String,
                          channels: Int)
  : RandomCropper = {
    if (cropperMethod == "Random") {
      RandomCropper(cropWidth, cropHeight, mirror,
        CropRandom, channels)
    } else {
      RandomCropper(cropWidth, cropHeight, mirror,
        CropCenter, channels)
    }
  }

  def createRandomResize(minSize: Int, maxSize : Int)
  : RandomResize = {
    RandomResize(minSize, maxSize)
  }

  def transformImageFeature(transformer: FeatureTransformer, feature: ImageFeature)
  : ImageFeature = {
    transformer.transform(feature)
  }

  def transformImageFrame(transformer: FeatureTransformer,
    imageFrame: ImageFrame): ImageFrame = {
    imageFrame.transform(transformer)
  }

  def setLabel(labelMap: JMap[String, Float], imageFrame: ImageFrame): Unit = {
    imageFrame.setLabel(labelMap.asScala)
  }

  def createDistributedImageFrame(imageRdd: JavaRDD[JTensor], labelRdd: JavaRDD[JTensor])
  : DistributedImageFrame = {
    require(null != imageRdd, "imageRdd cannot be null")
    val featureRdd = if (null != labelRdd) {
      imageRdd.rdd.zip(labelRdd.rdd).map(data => {
        createImageFeature(data._1, data._2)
      })
    } else {
      imageRdd.rdd.map(image => {
        createImageFeature(image, null)
      })
    }
    new DistributedImageFrame(featureRdd)
  }

  def createLocalImageFrame(images: JList[JTensor], labels: JList[JTensor])
  : LocalImageFrame = {
    require(null != images, "images cannot be null")
    val features = if (null != labels) {
      (0 until images.size()).map(i => {
        createImageFeature(images.get(i), labels.get(i))
      })
    } else {
      (0 until images.size()).map(i => {
        createImageFeature(images.get(i), null)
      })
    }
    new LocalImageFrame(features.toArray)
  }

  def createPipeline(list: JList[FeatureTransformer]): FeatureTransformer = {
    var cur = list.get(0)
    (1 until list.size()).foreach(t => cur = cur -> list.get(t))
    cur
  }


  def createImageFeature(data: JTensor = null, label: JTensor = null, uri: String = null)
  : ImageFeature = {
    val feature = new ImageFeature()
    if (null != data) {
      val mat = OpenCVMat.fromFloats(data.storage, data.shape(0), data.shape(1), data.shape(2))
      feature(ImageFeature.bytes) = OpenCVMat.imencode(mat)
      feature(ImageFeature.mat) = mat
      feature(ImageFeature.originalSize) = mat.shape()
    }
    if (null != label) {
      // todo: may need a method to change label format if needed
      feature(ImageFeature.label) = toTensor(label)
    }
    if (null != uri) {
      feature(ImageFeature.uri) = uri
    }
    feature
  }

  def imageFeatureGetKeys(imageFeature: ImageFeature): JList[String] = {
    imageFeature.keys().toList.asJava
  }

  def distributedImageFrameToImageTensorRdd(imageFrame: DistributedImageFrame,
    floatKey: String = ImageFeature.floats, toChw: Boolean = true): JavaRDD[JTensor] = {
    imageFrame.rdd.map(imageFeatureToImageTensor(_, floatKey, toChw)).toJavaRDD()
  }

  def distributedImageFrameToLabelTensorRdd(imageFrame: DistributedImageFrame): JavaRDD[JTensor] = {
    imageFrame.rdd.map(imageFeatureToLabelTensor).toJavaRDD()
  }

  def distributedImageFrameToPredict(imageFrame: DistributedImageFrame, key: String)
  : JavaRDD[JList[Any]] = {
    imageFrame.rdd.map(x => {
      if (x.isValid && x.contains(key)) {
        List[Any](x.uri(), toJTensor(x[Tensor[T]](key))).asJava
      } else {
        List[Any](x.uri(), null).asJava
      }
    })
  }

  def distributedImageFrameToSample(imageFrame: DistributedImageFrame, key: String):
  JavaRDD[Sample] = {
    imageFrame.rdd.map(x => {
      if (x.isValid && x.contains(key)) {
        toPySample(x[JSample[T]](key))
      } else {
        null
      }
    })
  }

  def distributedImageFrameToUri(imageFrame: DistributedImageFrame, key: String):
    JavaRDD[String] = {
    imageFrame.rdd.map(x => {
      if (x.contains(key)) {
        x[String](key)
      } else {
        null
      }
    })
  }

  def distributedImageFrameRandomSplit(imageFrame: DistributedImageFrame,
    weights: JList[Double]): Array[ImageFrame] = {
    return imageFrame.randomSplit(weights.asScala.toArray)
  }

  def localImageFrameToUri(imageFrame: LocalImageFrame, key: String): JList[String] = {
    imageFrame.array.map(x => {
      if (x.contains(key)) {
        x[String](key)
      } else {
        null
      }
    }).toList.asJava
  }

  def localImageFrameToSample(imageFrame: LocalImageFrame, key: String): JList[Sample] = {
    imageFrame.array.map(x => {
      if (x.isValid && x.contains(key)) {
        toPySample(x[JSample[T]](key))
      } else {
        null
      }
    }).toList.asJava
  }

  def localImageFrameToPredict(imageFrame: LocalImageFrame, key: String)
  : JList[JList[Any]] = {
    imageFrame.array.map(x =>
      if (x.isValid && x.contains(key)) {
        List[Any](x.uri(), toJTensor(x[Tensor[T]](key))).asJava
      } else {
        List[Any](x.uri(), null).asJava
      }).toList.asJava
  }

  def localImageFrameToImageTensor(imageFrame: LocalImageFrame,
    floatKey: String = ImageFeature.floats, toChw: Boolean = true): JList[JTensor] = {
    imageFrame.array.map(imageFeatureToImageTensor(_, floatKey, toChw)).toList.asJava
  }

  def localImageFrameToLabelTensor(imageFrame: LocalImageFrame): JList[JTensor] = {
    imageFrame.array.map(imageFeatureToLabelTensor).toList.asJava
  }

  def imageFeatureToImageTensor(imageFeature: ImageFeature,
    floatKey: String = ImageFeature.floats, toChw: Boolean = true): JTensor = {
    toJTensor(imageFeature.toTensor(floatKey, toChw).asInstanceOf[Tensor[T]])
  }

  def imageFeatureToLabelTensor(imageFeature: ImageFeature): JTensor = {
    val label = if (imageFeature.hasLabel()) {
      imageFeature.getLabel[Tensor[T]]
    } else {
      Tensor[T](1).fill(ev.fromType[Float](-1f))
    }
    toJTensor(label)
  }

  def read(path: String, sc: JavaSparkContext, minPartitions: Int): ImageFrame = {
    if (sc == null) {
      ImageFrame.read(path, null, minPartitions)
    } else {
      ImageFrame.read(path, sc.sc, minPartitions)
    }
  }

  def readParquet(path: String, sc: JavaSparkContext): DistributedImageFrame = {
    val sqlContext = new SQLContext(sc)
    ImageFrame.readParquet(path, sqlContext)
  }

  def writeParquet(path: String, output: String,
                   sc: JavaSparkContext, partitionNum: Int = 1): Unit = {
    val sqlContext = new SQLContext(sc)
    ImageFrame.writeParquet(path, output, sqlContext, partitionNum)
  }

  def createBytesToMat(byteKey: String): BytesToMat = {
    BytesToMat(byteKey)
  }

  def createPixelBytesToMat(byteKey: String): PixelBytesToMat = {
    PixelBytesToMat(byteKey)
  }

  def createMatToFloats(validHeight: Int = 300, validWidth: Int = 300, validChannels: Int = 3,
    outKey: String = ImageFeature.floats, shareBuffer: Boolean = true): MatToFloats =
    new MatToFloats(validHeight, validWidth, validChannels, outKey, shareBuffer)

  def createMatToTensor(toRGB: Boolean = false, tensorKey: String = ImageFeature.imageTensor)
  : MatToTensor[T] = new MatToTensor[T](toRGB, tensorKey)

  def isLocal(imageFrame: ImageFrame): Boolean = imageFrame.isLocal()

  def isDistributed(imageFrame: ImageFrame): Boolean = imageFrame.isDistributed()

  def createImageFrameToSample(inputKeys: JList[String],
    targetKeys: JList[String], sampleKey: String): ImageFrameToSample[T] = {
    val targets = if (targetKeys == null) null else targetKeys.asScala.toArray
    ImageFrameToSample[T](inputKeys.asScala.toArray, targets, sampleKey)
  }

  def seqFilesToImageFrame(url: String, sc: JavaSparkContext,
    classNum: Int, partitionNum: Int): ImageFrame = {
    val pn = if (partitionNum <= 0) None else Some(partitionNum)
    DataSet.SeqFileFolder.filesToImageFrame(url, sc, classNum, pn)
  }

  def setConstantClip(optimizer: Optimizer[T, MiniBatch[T]],
                      min: Float, max: Float): Unit = {
    optimizer.setConstantGradientClipping(min, max)
  }

  def setL2NormClip(optimizer: Optimizer[T, MiniBatch[T]],
                    normValue: Float): Unit = {
    optimizer.setGradientClippingByl2Norm(normValue)
  }

  def disableClip(optimizer: Optimizer[T, MiniBatch[T]]): Unit = {
    optimizer.disableGradientClipping()
  }

  def addScheduler(seq: SequentialSchedule, scheduler: LearningRateSchedule,
    maxIteration: Int): SequentialSchedule = {
    seq.add(scheduler, maxIteration)
  }

  private[bigdl] def initExecutorGateway(sc: JavaSparkContext, driverPort: Int): Unit = {
    sc.parallelize(Seq(""), Engine.coreNumber() * Engine.nodeNumber())
      .foreachPartition(_ => Engine.createJavaGateway(driverPort))
  }

  def createDatasetFromImageFrame(imageFrame: ImageFrame): DataSet[ImageFeature] = {
    DataSet.imageFrame(imageFrame)
  }

  def dlReadImage(path: String, sc: JavaSparkContext, minParitions: Int): DataFrame = {
    val df = DLImageReader.readImages(path, sc.sc, minParitions)
    df
  }

  def createDLImageTransformer(transformer: FeatureTransformer): DLImageTransformer = {
    new DLImageTransformer(transformer)
  }

  def dlImageTransform(dlImageTransformer: DLImageTransformer, dataSet: DataFrame): DataFrame = {
    dlImageTransformer.transform(dataSet)
  }

  def getRealClassNameOfJValue(module: AbstractModule[Activity, Activity, T]): String = {
    module.getClass.getCanonicalName
  }
}

object PythonBigDLUtils {
  def toTensor[T: ClassTag](jTensor: JTensor, typeName: String)
    (implicit ev: TensorNumeric[T]): Tensor[T] = {
    if (jTensor == null) return null

    typeName match {
      case "float" =>
        Tensor(jTensor.storage.map(x => ev.fromType(x.toFloat)), jTensor.shape)
      case "double" =>
        Tensor(jTensor.storage.map(x => ev.fromType(x.toDouble)), jTensor.shape)
      case t: String =>
        throw new IllegalArgumentException(s"Not supported type: ${t}")
    }
  }
}
