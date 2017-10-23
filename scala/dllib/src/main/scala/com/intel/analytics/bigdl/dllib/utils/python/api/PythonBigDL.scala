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
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, _}
import com.intel.analytics.bigdl.numeric._
import com.intel.analytics.bigdl.optim.{Optimizer, _}
import com.intel.analytics.bigdl.tensor.{DenseType, SparseType, Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Table, _}
import com.intel.analytics.bigdl.visualization.{Summary, TrainSummary, ValidationSummary}
import com.intel.analytics.bigdl.nn.Zeros
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.rdd.RDD
import java.lang.{Integer, Boolean => JBoolean}
import java.nio.ByteOrder

import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn.tf.{Const, Fill, Shape, SplitAndSelect}
import com.intel.analytics.bigdl.utils.tf.TensorflowLoader.{buildBigDLModel, buildTFGraph, parse}

import com.intel.analytics.bigdl.utils.tf.{BigDLSessionImpl, Context, TensorflowDataFormat, TensorflowSaver}

import org.apache.spark.ml.{DLClassifierModel, DLEstimator, DLClassifier, DLModel}
import org.apache.spark.sql.DataFrame

import org.apache.log4j._
import org.apache.spark.SparkContext
import org.tensorflow.framework.NodeDef

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.language.existentials
import scala.reflect.ClassTag


/**
 * [[com.intel.analytics.bigdl.dataset.Sample]] for python.
 * @param features features
 * @param label labels
 * @param bigdlType bigdl numeric type
 */
case class Sample(features: JList[JTensor],
                  label: JTensor,
                  bigdlType: String)

case class JTensor(storage: Array[Float], shape: Array[Int],
                   bigdlType: String, indices: Array[Array[Int]] = null)

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

  private def toTable(input: JList[JTensor]): Table = {
    input.asScala.foldLeft(new Table())((t, jtensor) => t.insert(toTensor(jtensor)))
  }

  def jTensorsToActivity(input: JList[JTensor], isTable: Boolean): Activity = {
    if (input.isEmpty) {
      throw new IllegalArgumentException("Empty input")
    }
    if (isTable) {
      toTable(input)
    } else {
      toTensor(input.iterator().next())
    }
  }

  def activityToJTensors(outputActivity: Activity): JList[JTensor] = {
    if (outputActivity.isInstanceOf[Tensor[T]]) {
      List(toJTensor(outputActivity.toTensor)).asJava
    } else {
      outputActivity.toTable.getState().toList.map {
        pair => (pair._1.asInstanceOf[Int], toJTensor(pair._2.asInstanceOf[Tensor[T]]))
      }.sortWith(_._1 < _._1).map(pair => pair._2).asJava
    }
  }

  def toPySample(sample: JSample[T]): Sample = {
    val cls = implicitly[ClassTag[T]].runtimeClass
    val features = new JArrayList[JTensor]()
    features.add(toJTensor(sample.feature()))
    Sample(features, toJTensor(sample.label()), cls.getSimpleName)
  }

  def toTensor(jTensor: JTensor): Tensor[T] = {
    if (jTensor == null) return null

    this.typeName match {
      case "float" =>
        if (null == jTensor.indices) {
          Tensor(jTensor.storage.map(x => ev.fromType(x)), jTensor.shape)
        } else {
          Tensor.sparse(jTensor.indices, jTensor.storage.map(x => ev.fromType(x)), jTensor.shape)
        }
      case "double" =>
        if (null == jTensor.indices) {
          Tensor(jTensor.storage.map(x => ev.fromType(x.toDouble)), jTensor.shape)
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
          JTensor(Array(), Array(0), bigdlType = typeName)
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
    val jsample = toSample(sample)
    toPySample(jsample)
  }

  def toSample(record: Sample): JSample[T] = {
    require(record.bigdlType == this.typeName,
      s"record.bigdlType: ${record.bigdlType} == this.typeName: ${this.typeName}")
    JSample[T](record.features.asScala.toArray.map(toTensor(_)), toTensor(record.label))
  }

  private def batching(rdd: RDD[Sample], batchSize: Int)
  : DistributedDataSet[MiniBatch[T]] = {
    val recordRDD = rdd.map(toSample(_))
    (DataSet.rdd(recordRDD) -> SampleToMiniBatch[T](batchSize))
      .asInstanceOf[DistributedDataSet[MiniBatch[T]]]
  }

  def createSequential(): Sequential[T] = {
    Sequential[T]()
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

  def createTimeDistributedCriterion(critrn: TensorCriterion[T],
    sizeAverage: Boolean = false): TimeDistributedCriterion[T] = {
    TimeDistributedCriterion[T](critrn, sizeAverage)
  }

  def createGRU(
    inputSize: Int,
    outputSize: Int,
    p: Double = 0,
    wRegularizer: Regularizer[T] = null,
    uRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null): GRU[T] = {
    GRU[T](inputSize, outputSize, p, wRegularizer, uRegularizer, bRegularizer)
  }

  def createLSTM(
    inputSize: Int,
    hiddenSize: Int,
    p: Double = 0,
    wRegularizer: Regularizer[T] = null,
    uRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null): LSTM[T] = {
    LSTM[T](inputSize, hiddenSize, p, wRegularizer, uRegularizer, bRegularizer)
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
    wRegularizer: Regularizer[T] = null,
    uRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    cRegularizer: Regularizer[T] = null,
    withPeephole: Boolean = true): ConvLSTMPeephole[T] = {
    ConvLSTMPeephole[T](inputSize, outputSize, kernelI, kernelC, stride,
      wRegularizer, uRegularizer, bRegularizer, cRegularizer, withPeephole)
  }

  def createConvLSTMPeephole3D(
    inputSize: Int,
    outputSize: Int,
    kernelI: Int,
    kernelC: Int,
    stride: Int = 1,
    wRegularizer: Regularizer[T] = null,
    uRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    cRegularizer: Regularizer[T] = null,
    withPeephole: Boolean = true): ConvLSTMPeephole3D[T] = {
    ConvLSTMPeephole3D[T](inputSize, outputSize, kernelI, kernelC, stride,
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
    initGradBias: JTensor = null)
  : SpatialBatchNormalization[T] = {
    SpatialBatchNormalization[T](nOutput, eps, momentum, affine,
      toTensor(initWeight), toTensor(initBias), toTensor(initGradWeight), toTensor(initBias))
  }

  def createSpatialCrossMapLRN(size: Int = 5,
    alpha: Double = 1.0,
    beta: Double = 0.75,
    k: Double = 1.0)
  : SpatialCrossMapLRN[T] = {
    SpatialCrossMapLRN[T](size, alpha, beta, k)
  }

  def createDropout(initP: Double = 0.5,
    inplace: Boolean = false,
    scale: Boolean = true)
  : Dropout[T] = {
    Dropout[T](initP, inplace, scale)
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
  : CAddTable[T] = {
    CAddTable[T](inplace)
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

  def createL1Penalty(l1weight: Int,
    sizeAverage: Boolean = false,
    provideOutput: Boolean = true)
  : L1Penalty[T] = {
    L1Penalty[T](l1weight,
      sizeAverage,
      provideOutput)
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
  : Mean[T, T] = {
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

  def createSigmoid()
  : Sigmoid[T] = {
    Sigmoid[T]()
  }

  def createSoftMax()
  : SoftMax[T] = {
    SoftMax[T]()
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
  : Sum[T, T] = {
    Sum[T, T](dimension,
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

  def createUnsqueeze(pos: Int,
    numInputDims: Int = Int.MinValue)
  : Unsqueeze[T] = {
    Unsqueeze[T](pos,
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

  def createClassNLLCriterion(weights: JTensor = null,
    sizeAverage: Boolean = true)
  : ClassNLLCriterion[T] = {
    ClassNLLCriterion[T](if (weights == null) null else toTensor(weights),
      sizeAverage)
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
    sizeAverage: Boolean = true)
  : MarginCriterion[T] = {
    MarginCriterion[T](margin,
      sizeAverage)
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

  def createKLDCriterion(): KLDCriterion[T] = {
    KLDCriterion[T]()
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

  def createPack(dimension: Int): Pack[T] = {
    Pack(dimension)
  }

  def setModelSeed(seed: Long): Unit = {
    RandomGenerator.RNG.setSeed(seed)
  }

  def modelEvaluate(model: AbstractModule[Activity, Activity, T],
    valRDD: JavaRDD[Sample],
    batchSize: Int,
    valMethods: JList[ValidationMethod[T]])
  : JList[EvaluatedResult] = {
    val resultArray = model.evaluate(valRDD.rdd.map(toSample(_)),
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

  def loadBigDLModule(path: String): AbstractModule[Activity, Activity, T] = {
    Module.loadModule[T](path)
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
    byteOrder: String): AbstractModule[Activity, Activity, T] = {
    val order = byteOrder match {
      case "little_endian" => ByteOrder.LITTLE_ENDIAN
      case "big_endian" => ByteOrder.BIG_ENDIAN
      case _ => throw new IllegalArgumentException(s"No support byte order $byteOrder")
    }
    Module.loadTF[T](path, inputs.asScala, outputs.asScala, order)
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

  def modelPredictRDD(model: AbstractModule[Activity, Activity, T],
    dataRdd: JavaRDD[Sample]): JavaRDD[JTensor] = {
    val tensorRDD = model.predict(dataRdd.rdd.map(toSample(_)))
    val listRDD = tensorRDD.map { res =>
      val tensor = res.asInstanceOf[Tensor[T]]
      val cloneTensor = tensor.clone()
      toJTensor(cloneTensor)

    }
    new JavaRDD[JTensor](listRDD)
  }

  def evaluate(module: AbstractModule[Activity, Activity, T]):
  AbstractModule[Activity, Activity, T] = {
    module.evaluate()
  }

  def modelPredictClass(model: AbstractModule[Activity, Activity, T],
    dataRdd: JavaRDD[Sample]): JavaRDD[Int] = {
    val tensorRDD = model.predictClass(dataRdd.rdd.map(toSample(_)))
    new JavaRDD[Int](tensorRDD)
  }

  def modelForward(model: AbstractModule[Activity, Activity, T],
    input: JList[JTensor],
    inputIsTable: Boolean): JList[JTensor] = {
    val inputActivity = jTensorsToActivity(input, inputIsTable)
    val outputActivity = model.forward(inputActivity)
    activityToJTensors(outputActivity)
  }

  def modelBackward(model: AbstractModule[Activity, Activity, T],
    input: JList[JTensor],
    inputIsTable: Boolean,
    gradOutput: JList[JTensor],
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
    path: String, overWrite: Boolean): Unit = {
    module.saveModule(path, overWrite)
  }

  def saveCaffe(module: AbstractModule[Activity, Activity, T],
    prototxtPath: String, modelPath: String,
    useV2: Boolean = true, overwrite: Boolean = false): Unit = {
    module.saveCaffe(prototxtPath, modelPath, useV2, overwrite)
  }

  def criterionForward(criterion: AbstractCriterion[Activity, Activity, T],
    input: JList[JTensor],
    inputIsTable: Boolean,
    target: JList[JTensor],
    targetIsTable: Boolean): T = {
    val inputActivity = jTensorsToActivity(input, inputIsTable)
    val targetActivity = jTensorsToActivity(target, targetIsTable)
    return criterion.forward(inputActivity, targetActivity)
  }

  def criterionBackward(criterion: AbstractCriterion[Activity, Activity, T],
    input: JList[JTensor],
    inputIsTable: Boolean,
    target: JList[JTensor],
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

  def createTop1Accuracy(): ValidationMethod[T] = {
    new Top1Accuracy()
  }

  def createTreeNNAccuracy(): ValidationMethod[T] = {
    new TreeNNAccuracy()
  }

  def createTop5Accuracy(): ValidationMethod[T] = {
    new Top5Accuracy()
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
    val dataset = batching(samples, batchSize)

    val model = session.train(Seq(output), dataset,
      optMethod, criterion, endWhen)
    model
  }

  def createOptimizer(model: AbstractModule[Activity, Activity, T],
    trainingRdd: JavaRDD[Sample],
    criterion: Criterion[T],
    optimMethod: OptimMethod[T],
    endTrigger: Trigger,
    batchSize: Int): Optimizer[T, MiniBatch[T]] = {
    val optimizer = new DistriOptimizer(
      _model = model,
      dataset = batching(trainingRdd, batchSize),
      criterion = criterion
    ).asInstanceOf[Optimizer[T, MiniBatch[T]]]
    // TODO: we should provide a more convenient way to create Table

    optimizer.setEndWhen(endTrigger)

    optimizer.setOptimMethod(optimMethod)

    // TODO: remove this
    optimizer.disableCheckSingleton()

    optimizer
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
    optimizer.setValidation(trigger, batching(valRdd, batchSize.toInt), vMethods.asScala.toArray)
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

  def createModel(input: JList[ModuleNode[T]], output: JList[ModuleNode[T]]): Graph[T] = {
    Graph(input.asScala.toArray, output.asScala.toArray)
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
    model.updateParameters(ev.fromType(lr))
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

  def createBilinearFiller(): BilinearFiller.type = {
    BilinearFiller
  }

  def setInitMethod(layer: Initializable, weightInitMethod: InitializationMethod,
    biasInitMethod: InitializationMethod): layer.type = {
    layer.setInitMethod(weightInitMethod, biasInitMethod)
  }

  def getHiddenStates(rec: Recurrent[T]): JList[JTensor] = {
    val states = rec.getHiddenState()
    activityToJTensors(states)
  }

  def setHiddenStates(rec: Recurrent[T], hiddenStates: JList[JTensor], isTable: Boolean): Unit = {
      rec.setHiddenState(jTensorsToActivity(hiddenStates, isTable))
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

  def createResizeBilinear(
    outputHeight: Int,
    outputWidth: Int,
    alignCorner: Boolean
  ): ResizeBilinear[T] = {
    ResizeBilinear[T](outputHeight,
      outputWidth,
      alignCorner)
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
