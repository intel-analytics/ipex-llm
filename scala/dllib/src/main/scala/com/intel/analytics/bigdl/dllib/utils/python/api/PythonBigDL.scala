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
import com.intel.analytics.bigdl.dataset.{Sample => JSample, _}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorCriterion, TensorModule}
import com.intel.analytics.bigdl.numeric._
import com.intel.analytics.bigdl.optim.{Optimizer, _}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils._
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.rdd.RDD
import java.lang.Integer
import scala.collection.JavaConverters._
import scala.language.existentials
import scala.reflect.ClassTag

case class Sample(features: JList[Any],
                  label: JList[Any],
                  featuresShape: JList[Int],
                  labelShape: JList[Int],
                  bigdlType: String)

case class JTensor(storage: JList[Any], shape: JList[Int], bigdlType: String)

case class TestResult(val result: Float, totalNum: Int, val method: String)


object PythonBigDL {
  val floatInstance = new PythonBigDL[Float]()

  val doubleInstance = new PythonBigDL[Double]()

  def ofFloat(): PythonBigDL[Float] = floatInstance

  def ofDouble(): PythonBigDL[Double] = doubleInstance

  def getInitMethod(initMethod: String): InitializationMethod = {
    initMethod.toLowerCase() match {
      case "xavier" => Xavier
      case "default" => Default
      case "bilinearfiller" => BilinearFiller
      case m: String => throw new IllegalArgumentException(s"Not supported init method: ${m}")
    }
  }
}

class PythonBigDL[T: ClassTag](implicit ev: TensorNumeric[T]) extends Serializable {

  private val typeName = {
    val cls = implicitly[ClassTag[T]].runtimeClass
    cls.getSimpleName
  }


  private def toValidationMethod(vMethods: JList[String]): Array[ValidationMethod[T]] = {
    vMethods.toArray.map {
      case "Top1Accuracy" => new Top1Accuracy[T]()
      case "Top5Accuracy" => new Top5Accuracy[T]()
      case "Loss" => new Loss[T]()
      case m: String => throw new RuntimeException(s"not supported validation method: $m")
    }
  }

  private def validationMethodToStr(method: ValidationMethod[T]): String = {
    method match {
      case _: Top1Accuracy[T] => "Top1Accuracy"
      case _: Top5Accuracy[T] => "Top5Accuracy"
      case _: Loss[T] => "loss"
      case _ => throw new RuntimeException(s"not supported validation method: $method")
    }
  }

  def toPySample(sample: JSample[T]): Sample = {
    val featureList = sample.feature().contiguous().storage().toArray[T].toList.asJava
    val labelList = sample.label().contiguous().storage().toArray[T].toList.asJava
    val cls = implicitly[ClassTag[T]].runtimeClass
    Sample(featureList.asInstanceOf[JList[Any]],
      labelList.asInstanceOf[JList[Any]],
      sample.feature().size().toList.asJava,
      sample.label().size().toList.asJava,
      cls.getSimpleName)
  }

  def toTensor(jTensor: JTensor): Tensor[T] = {
    Tensor(jTensor.storage.asScala.toArray.asInstanceOf[Array[T]],
      jTensor.shape.asScala.toArray)
  }

  def toJTensor(tensor: Tensor[T]): JTensor = {
    // TODO: we should clone here, in case the underlying storage large than the tensor size.
    val cloneTensor = tensor
    JTensor(cloneTensor.storage().toList.asJava.asInstanceOf[JList[Any]],
      cloneTensor.size().toList.asJava, typeName)
  }

  def testTensor(jTensor: JTensor): JTensor = {
    val tensor = toTensor(jTensor)
    toJTensor(tensor)
  }

  def toSample(record: Sample): JSample[T] = {
    require(record.bigdlType == this.typeName,
      s"record.bigdlType: ${record.bigdlType} == this.typeName: ${this.typeName}")
    val sample = this.typeName match {
      case "float" =>
        JSample[Float]().set(
          record.features.asInstanceOf[JList[Double]].asScala.map(_.toFloat).toArray[Float],
          (record.label.asInstanceOf[JList[Double]]).asScala.map(_.toFloat).toArray[Float],
          record.featuresShape.asScala.toArray[Int],
          record.labelShape.asScala.toArray[Int])
      case "double" =>
        JSample[Double]().set(
          record.features.asInstanceOf[JList[Double]].asScala.toArray[Double],
          (record.label.asInstanceOf[JList[Double]]).asScala.toArray[Double],
          record.featuresShape.asScala.toArray[Int],
          record.labelShape.asScala.toArray[Int])
      case t: String =>
        throw new IllegalArgumentException(s"Not supported type: ${t}")
    }
    sample.asInstanceOf[JSample[T]]
  }

  private def batching(rdd: RDD[Sample], batchSize: Int)
  : DistributedDataSet[MiniBatch[T]] = {
    val recordRDD = rdd.map(toSample(_))
    (DataSet.rdd(recordRDD) -> new SampleToBatch[T](batchSize))
      .asInstanceOf[DistributedDataSet[MiniBatch[T]]]
  }

  def createSequential(): Sequential[T] = {
    Sequential[T]()
  }

  def createLinear(inputSize: Int, outputSize: Int,
                   initMethod: String, withBias: Boolean): Linear[T] = {
    Linear[T](inputSize, outputSize, PythonBigDL.getInitMethod(initMethod), withBias)
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

  def createRNNCell(inputSize: Int,
                    hiddenSize: Int,
                    activation: TensorModule[T]): RnnCell[T] = {
    RnnCell[T](inputSize, hiddenSize, activation)
  }

  def createTimeDistributedCriterion(critrn: TensorCriterion[T],
                                     sizeAverage: Boolean = false): TimeDistributedCriterion[T] = {
    TimeDistributedCriterion[T](critrn, sizeAverage)
  }

  def createGRU(inputSize: Int,
                outputSize: Int): GRU[T] = {
    GRU[T](inputSize, outputSize)
  }

  def createLSTM(inputSize: Int,
                 hiddenSize: Int): LSTM[T] = {
    LSTM[T](inputSize, hiddenSize)
  }

  def createRecurrent(): Recurrent[T] = {
    Recurrent[T]()
  }

  def createEcho(): Echo[T] = {
    Echo[T]()
  }

  def createLogSoftMax(): LogSoftMax[T] = {
    LogSoftMax[T]()
  }

  def createSpatialMaxPooling(kW: Int,
                              kH: Int,
                              dW: Int,
                              dH: Int,
                              padW: Int = 0,
                              padH: Int = 0,
                              ceilMode: Boolean = false)
  : SpatialMaxPooling[T] = {
    val maxpooling = SpatialMaxPooling[T](kW,
      kH,
      dW,
      dH,
      padW,
      padH)
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
                               initMethod: String = "default")
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
      PythonBigDL.getInitMethod(initMethod))
  }

  def createReshape(size: JList[Int]): Reshape[T] = {
    Reshape(size.asScala.toArray)
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
                                  ceilMode: Boolean = false,
                                  countIncludePad: Boolean = true,
                                  divide: Boolean = true)
  : SpatialAveragePooling[T] = {
    SpatialAveragePooling[T](kW, kH, dW, dH, padW, padH, ceilMode, countIncludePad, divide)
  }

  def createSpatialBatchNormalization(nOutput: Int,
                                      eps: Double = 1e-5,
                                      momentum: Double = 0.1,
                                      affine: Boolean = true)
  : SpatialBatchNormalization[T] = {
    SpatialBatchNormalization[T](nOutput, eps, momentum, affine)
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

  def createView(sizes: JList[Int], num_input_dims: Int): View[T] = {
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
                               affine: Boolean = true)
  : BatchNormalization[T] = {
    BatchNormalization[T](nOutput,
      eps,
      momentum,
      affine)
  }

  def createBilinear(inputSize1: Int,
                     inputSize2: Int,
                     outputSize: Int,
                     biasRes: Boolean = true)
  : Bilinear[T] = {
    Bilinear[T](inputSize1,
      inputSize2,
      outputSize,
      biasRes)
  }

  def createBottle(module: AbstractModule[Activity, Activity, T],
                   nInputDim: Int = 2,
                   nOutputDim1: Int = Int.MaxValue)
  : Bottle[T] = {
    Bottle[T](module,
      nInputDim,
      nOutputDim1)
  }

  def createCAdd(size: JList[Int])
  : CAdd[T] = {
    CAdd[T](size.asScala.toArray)
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

  def createCMul(size: JList[Int])
  : CMul[T] = {
    CMul[T](size.asScala.toArray)
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

  def createCopy()
  : Copy[T] = {
    Copy[T]()
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
                        normType: Double = 2.0, shouldScaleGradByFreq: Boolean = false)
  : LookupTable[T] = {
    LookupTable[T](nIndex,
      nOutput,
      paddingValue,
      maxNorm,
      normType,
      shouldScaleGradByFreq)
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
                 nInputDims: Int = -1)
  : Mean[T] = {
    Mean[T](dimension,
      nInputDims)
  }

  def createMin(dim: Int = 1,
                numInputDims: Int = Int.MinValue)
  : Min[T] = {
    Min[T](dim,
      numInputDims)
  }

  def createMixtureTable()
  : MixtureTable[T] = {
    MixtureTable[T]()
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

  def createRoiPooling(pooled_w: Int, pooled_h: Int, spatial_scale: T)
  : RoiPooling[T] = {
    RoiPooling[T](pooled_w,
      pooled_h,
      spatial_scale)
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
                                      initMethod: String = "default")
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
      PythonBigDL.getInitMethod(initMethod))
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
                                   initMethod: String = "default")
  : SpatialFullConvolution[Activity, T] = {
    SpatialFullConvolution[Activity, T](nInputPlane,
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
      PythonBigDL.getInitMethod(initMethod))
  }

  def createSpatialShareConvolution(nInputPlane: Int,
                                    nOutputPlane: Int,
                                    kernelW: Int,
                                    kernelH: Int,
                                    strideW: Int = 1,
                                    strideH: Int = 1,
                                    padW: Int = 0,
                                    padH: Int = 0,
                                    nGroup: Int = 1,
                                    propagateBack: Boolean = true,
                                    initMethod: String = "default")
  : SpatialShareConvolution[T] = {
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
      PythonBigDL.getInitMethod(initMethod))
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
                sizeAverage: Boolean = false)
  : Sum[T] = {
    Sum[T](dimension,
      nInputDims,
      sizeAverage)
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

  //   Optimizer
  def createPoly(power: Double, maxIteration: Int): SGD.Poly = {
    SGD.Poly(power, maxIteration)
  }

  def createStep(stepSize: Int, gamma: Double): SGD.Step = {
    SGD.Step(stepSize, gamma)
  }

  def createClassNLLCriterion(sizeAverage: Boolean = true)
  : ClassNLLCriterion[T] = {
    ClassNLLCriterion[T](null,
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

  def setModelSeed(seed: Long): Unit = {
    RandomGenerator.RNG.setSeed(seed)
  }

  def modelTest(model: AbstractModule[Activity, Activity, T],
                valRDD: JavaRDD[Sample],
                batchSize: Int,
                valMethods: JList[String])
  : JList[TestResult] = {
    val validator = Validator(model, batching(valRDD, batchSize))
    val resultArray = validator.test(toValidationMethod(valMethods))
    val testResultArray = resultArray.map { result =>
      TestResult(result._1.result()._1, result._1.result()._2,
        validationMethodToStr(result._2))
    }
    testResultArray.toList.asJava
  }

  def loadBigDL(path: String): AbstractModule[Activity, Activity, T] = {
    Module.load[T](path)
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

  def modelPredictRDD(model: AbstractModule[Activity, Activity, T],
                      dataRdd: JavaRDD[Sample]): JavaRDD[Sample] = {
    val result = predict(model, dataRdd.rdd.map(toSample(_)))
    result.map(toPySample(_))

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

  def predict(model: AbstractModule[Activity, Activity, T],
              dataRdd: RDD[JSample[T]]): RDD[JSample[T]] = {
    val modelBroadCast = dataRdd.sparkContext.broadcast(model.evaluate())
    dataRdd.mapPartitions { partition =>
      val localModel = modelBroadCast.value.cloneModule()
      partition.map { sample =>
        val output = localModel.forward(sample.feature()).toTensor[T]
        JSample(sample.feature(), output)
      }
    }
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


  def createOptimizer(model: AbstractModule[Activity, Activity, T],
                      trainingRdd: JavaRDD[Sample],
                      criterion: Criterion[T],
                      optimMethod: String,
                      state: JMap[Any, Any],
                      endTrigger: Trigger,
                      batchSize: Int): Optimizer[T, MiniBatch[T]] = {
    val optimizer = new DistriOptimizer(
      model = model,
      dataset = batching(trainingRdd, batchSize),
      criterion = criterion
    ).asInstanceOf[Optimizer[T, MiniBatch[T]]]
    // TODO: we should provide a more convenient way to create Table
    val stateTable = new Table()
    state.asScala.foreach { case (key, value) =>
      stateTable.update(key, value)
    }
    optimizer.setState(stateTable)

    optimizer.setEndWhen(endTrigger)

    optimMethod.toLowerCase() match {
      case "sgd" =>
        optimizer.setOptimMethod(new SGD())
      case "adagrad" =>
        optimizer.setOptimMethod(new Adagrad())
      case "lbfgs" =>
        optimizer.setOptimMethod(new LBFGS())
      case n: String => throw new IllegalArgumentException(s"Not supported type: $n")
    }
    // TODO: remove this
    optimizer.disableCheckSingleton()

    optimizer
  }

  def setValidation(optimizer: Optimizer[T, MiniBatch[T]],
                    batchSize: Int,
                    trigger: Trigger,
                    valRdd: JavaRDD[Sample],
                    vMethods: JList[String]): Unit = {
    optimizer.setValidation(trigger, batching(valRdd, batchSize.toInt),
      toValidationMethod(vMethods))
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


  def initEngine(): Unit = {
    Engine.init
  }
}



