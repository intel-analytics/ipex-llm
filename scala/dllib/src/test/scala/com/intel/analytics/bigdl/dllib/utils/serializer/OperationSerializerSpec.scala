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

package com.intel.analytics.bigdl.utils.serializer

import java.io.{File => JFile}

import com.google.protobuf.{ByteString, CodedOutputStream}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.nn.ops.{All, Any, ApproximateEqual, ArgMax, BatchMatMul, BucketizedCol, Cast, CategoricalColHashBucket, CategoricalColVocaList, Ceil, CrossCol, CrossEntropy, DepthwiseConv2D, DepthwiseConv2DBackpropFilter, DepthwiseConv2DBackpropInput, Digamma, Dilation2D, Dilation2DBackpropFilter, Dilation2DBackpropInput, Equal, Erf, Erfc, Expm1, Floor, FloorDiv, FloorMod, Greater, GreaterEqual, InTopK, IndicatorCol, Inv, InvGrad, IsFinite, IsInf, IsNan, Kv2Tensor, L2Loss, Less, LessEqual, Lgamma, LogicalAnd, LogicalNot, LogicalOr, Maximum, Minimum, Mod, ModuleToOperation, NotEqual, OneHot, Pad, Prod, RandomUniform, RangeOps, Rank, ResizeBilinearGrad, ResizeBilinearOps, Rint, Round, SegmentSum, SelectTensor, Sign, Slice, SquaredDifference, Substr, TensorOp, TopK, TruncateDiv, TruncatedNormal, Exp => ExpOps, Pow => PowOps, Select => SelectOps, Sum => SumOps, Tile => TileOps}
import com.intel.analytics.bigdl.nn.tf.{Assert => AssertOps, BroadcastGradientArgs => BroadcastGradientArgsOps, DecodeGif => DecodeGifOps, DecodeJpeg => DecodeJpegOps, DecodePng => DecodePngOps, DecodeRaw => DecodeRawOps}
import com.intel.analytics.bigdl.nn.tf._
import com.intel.analytics.bigdl.nn.{SoftPlus => BigDLSoftPlus}
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.tf.TFRecordIterator
import com.intel.analytics.bigdl.utils.tf.loaders.{Pack => _, _}
import org.tensorflow.example._
import org.tensorflow.framework.DataType

import scala.collection.mutable
import scala.util.Random

class OperationSerializerSpec extends SerializerSpecHelper {

  override protected def getPackage(): String = "com.intel.analytics.bigdl.nn.ops"

  override def addExcludedPackage(): Unit = {
    excludedPackage.add("com.intel.analytics.bigdl.utils.tf.loaders")
    excludedPackage.add("com.intel.analytics.bigdl.utils.tf.ops")
    // It would be tested in a separated spec
    excludedPackage.add("com.intel.analytics.bigdl.nn.keras")
  }

  override def getExpected(): mutable.Set[String] = {
    super.getExpected().filter(cls => {
      cls.contains(getPackage()) || cls.contains("com.intel.analytics.bigdl.tf")
    })
  }

  "All serializer" should "work properly" in {
    val all = All[Float]().setName("all")
    val input1 = Tensor[Boolean](T(T(true, true, false), T(false, true, true)))
    val input2 = Tensor[Int](T(2, 1, 2))
    val input = T()
    input(1.toFloat) = input1
    input(2.toFloat) = input2
    runSerializationTest(all, input)
  }

  "Any serializer" should "work properly" in {
    val any = Any[Float]().setName("any")
    val input1 = Tensor[Boolean](T(T(true, true, false), T(false, true, true)))
    val input2 = Tensor[Int](T(2, 1, 2))
    val input = T()
    input(1.toFloat) = input1
    input(2.toFloat) = input2
    runSerializationTest(any, input)
  }

  "ApproximateEqual serializer" should "work properly" in {
    val approximateEqual = ApproximateEqual[Float](0.01f).setName("approximateEqual")
    val input = T(Tensor[Float](5).apply1(_ => Random.nextFloat()),
      Tensor[Float](5).apply1(_ => Random.nextFloat()))
    runSerializationTest(approximateEqual, input, approximateEqual.
      asInstanceOf[ModuleToOperation[Float]].module.getClass
    )
  }

  "ArgMax serializer" should "work properly" in {
    val argMax = ArgMax[Float].setName("argMax")
    val dataTensor = Tensor[Float](T(T(1.0f, 2.0f), T(3.0f, 4.0f)))
    val dimensionTensor = Tensor.scalar[Int](1)
    val input = T(dataTensor, dimensionTensor)
    runSerializationTest(argMax, input)
  }

  "Assert serializer" should "work properly" in {
    import com.intel.analytics.bigdl.utils.tf.TFTensorNumeric.NumericByteString
    val assert = new AssertOps[Float]().setName("assert")
    val predictTensor = Tensor[Boolean](Array(1))
    predictTensor.setValue(1, true)
    val msg = Tensor[ByteString](Array(1))
    msg.setValue(1, ByteString.copyFromUtf8("must be true"))
    val input = T(predictTensor, msg)
    runSerializationTest(assert, input)
  }

  "Assign serializer" should "work properly" in {
    val assign = new Assign[Float]().setName("assign")
    val input =
      T(
        Tensor[Float](T(1f, 2f, 3f)),
        Tensor[Float](T(2f, 2f, 4f))
      )
    runSerializationTest(assign, input)
  }

  "AssignGrad serializer" should "work properly" in {
    val grad = Tensor[Float](5).apply1(_ => Random.nextFloat())
    val assignGrad = new AssignGrad[Float](grad).setName("assignGrad")
    val input = Tensor[Float](5).apply1(_ => Random.nextFloat())
    runSerializationTest(assignGrad, input)
  }

  "AvgPoolGrad serializer" should "work properly" in {
    val avgPoolGrad = AvgPoolGrad[Float](4, 4, 1, 1, -1, -1, DataFormat.NHWC).
      setName("avgPoolGrad")
    val input1 = Tensor[Int](T(4, 32, 32, 3))
    val input2 = Tensor[Float](4, 32, 32, 3).apply1(_ => Random.nextFloat())
    val input = T(input1, input2)
    runSerializationTest(avgPoolGrad, input)
  }

  "BatchMatMul serializer" should "work properly" in {
    val batchMatMul = BatchMatMul[Float, Float]().setName("batchMatMul")
    val input =
      T(
        Tensor[Float](2, 2).apply1(_ => Random.nextFloat()),
        Tensor[Float](2, 2).apply1(_ => Random.nextFloat())
      )
    runSerializationTest(batchMatMul, input)
  }

  "BiasAddGrad serializer" should "work properly" in {
    val biasAddGrad = BiasAddGrad[Float](DataFormat.NCHW).
      setName("biasAddGrad")
    val input = Tensor[Float](2, 2, 2).apply1(_ => Random.nextFloat())
    runSerializationTest(biasAddGrad, input)
  }

  "BroadcastGradientArgs serializer" should "work properly" in {
    val broadcastGradientArgs = BroadcastGradientArgsOps[Float]().
      setName("broadcastGradientArgs")
    val input =
      T(
        Tensor[Int](T(1, 2, 3)),
        Tensor[Int](T(2, 2, 1))
      )
    runSerializationTest(broadcastGradientArgs, input, broadcastGradientArgs.
      asInstanceOf[ModuleToOperation[Float]].module.getClass)
  }

  "BucketizedCol serializer" should "work properly" in {
    val bucketizedCol = BucketizedCol[Float](boundaries = Array(0.0, 10.0, 100.0))
      .setName("bucketizedCol")
    val input = Tensor[Float](T(T(-1, 1), T(101, 10), T(5, 100)))
    runSerializationTest(bucketizedCol, input)
  }

  "Cast serializer" should "work properly" in {
    val cast = Cast[Float, Float]().setName("cast")
    val input = Tensor[Float](2, 2).apply1(_ => Random.nextFloat())
    runSerializationTest(cast, input, cast.
      asInstanceOf[ModuleToOperation[Float]].module.getClass)
  }

  "Ceil serializer" should "work properly" in {
    val ceil = Ceil[Float, Float]().setName("ceil")
    val input = Tensor[Float](2, 2).apply1(_ => Random.nextFloat())
    runSerializationTest(ceil, input)
  }

  "MergeOps serializer" should "work properly" in {
    val mergeOps = new MergeOps[Float](1).setName("mergeOps")
    val input =
      T(
        Tensor[Float](T(1.0f, 2.0f, 3.0f)),
        Tensor[Float](T(2.0f, 2.0f, 1.0f))
      )
    runSerializationTest(mergeOps, input)
  }

  "SwitchOps serializer" should "work properly" in {
    val switchOps = new SwitchOps[Float]().setName("switchOps")
    val input =
      T(
        Tensor[Float](T(1.0f, 2.0f, 3.0f)),
        Tensor[Boolean](T(true))
      )
    runSerializationTest(switchOps, input)
  }

  "Conv2D serializer" should "work properly" in {
    val conv2d = Conv2D[Float](2, 1, -1, -1).setName("conv2d")
    val inputTensor = Tensor[Float](1, 4, 3, 3).apply1(_ => Random.nextFloat())
    val filter = Tensor[Float](4, 3, 3, 2).apply1(_ => Random.nextFloat())
    val input = T(inputTensor, filter)
    runSerializationTest(conv2d, input)
  }

  "Conv2DBackFilter serializer" should "work properly" in {
    val conv2dBackFilter = Conv2DBackFilter[Float](2, 2, -1, -1, DataFormat.NHWC).
      setName("conv2dBackFilter")
    val inputTensor = Tensor[Float](1, 4, 3, 3).apply1(_ => Random.nextFloat())
    val kernelSize = Tensor[Int](T(2, 2, 3, 3))
    val grad = Tensor[Float](1, 2, 2, 3).apply1(_ => Random.nextFloat())
    val input = T(inputTensor, kernelSize, grad)
    runSerializationTest(conv2dBackFilter, input)
  }

  "Conv2DTranspose Serializer" should "work properly" in {
    val conv2dTranspose = Conv2DTranspose[Float](2, 2, -1, -1, DataFormat.NHWC).
      setName("conv2dTranspose")
    val inputTensor = Tensor[Int](T(1, 4, 3, 3))
    val kernelSize = Tensor[Float](2, 2, 3, 3).apply1(_ => Random.nextFloat())
    val data = Tensor[Float](1, 2, 2, 3)apply1(_ => Random.nextFloat())
    val input = T(inputTensor, kernelSize, data)
    runSerializationTest(conv2dTranspose, input)
  }

  "CrossCol Serializer" should "work proprly" in {
    val crosscol = CrossCol[Float](hashBucketSize = 100)
      .setName("CrossCol")
    val input = T(
      Tensor[String](T("A,D", "B", "A,C")),
      Tensor[String](T("1", "2", "3,4"))
    )
    runSerializationTest(crosscol, input)
  }

  "CrossEntropy serializer" should "work properly" in {
    val crossEntropy = CrossEntropy[Float]().setName("crossEntropy")
    val output = Tensor[Float](2, 5).apply1(_ => Random.nextFloat())
    val label = Tensor[Float](2, 5).apply1(_ => Random.nextFloat())
    val input = T(output, label)
    runSerializationTest(crossEntropy, input)
  }

  private def getInputs(name: String): Tensor[ByteString] = {
    import com.intel.analytics.bigdl.utils.tf.TFTensorNumeric.NumericByteString
    val index = name match {
      case "png" => 0
      case "jpeg" => 1
      case "gif" => 2
      case "raw" => 3
    }

    val resource = getClass.getClassLoader.getResource("tf")
    val path = resource.getPath + JFile.separator + "decode_image_test_case.tfrecord"
    val file = new JFile(path)

    val bytesVector = TFRecordIterator(file).toVector
    val pngBytes = bytesVector(index)

    val example = Example.parseFrom(pngBytes)
    val imageByteString = example.getFeatures.getFeatureMap.get("image/encoded")
      .getBytesList.getValueList.get(0)

    Tensor[ByteString](Array(imageByteString), Array[Int]())
  }

  "DecodeImage Serializer" should "work properly" in {
    val decodeImage = new DecodeImage[Float](1).setName("decodeImage")
    val input = getInputs("png")
    runSerializationTest(decodeImage, input)
  }

  "DecodeGif Serializer" should "work properly" in {
    val decodeGif = new DecodeGifOps[Float]().setName("decodeGif")
    val input = getInputs("gif")
    runSerializationTest(decodeGif, input)
  }

  "DecodeJpeg Serializer" should "work properly" in {
    val decodeJpeg = new DecodeJpegOps[Float](1).setName("decodeJpeg")
    val input = getInputs("jpeg")
    runSerializationTest(decodeJpeg, input)
  }

  "DecodePng Serializer" should "work properly" in {
    val decodePng = new DecodePngOps[Float](1).setName("decodePng")
    val input = getInputs("png")
    runSerializationTest(decodePng, input)
  }


  "DecodeRaw Serializer" should "work properly" in {
    val decodeRaw = new DecodeRawOps[Float](DataType.DT_UINT8, true).setName("decodeRaw")
    val input = getInputs("raw")
    runSerializationTest(decodeRaw, input)
  }

  "DepthwiseConv2DBackpropInput serializer" should "work properly" in {
    val depWiseBackprop =
      DepthwiseConv2DBackpropInput[Float](1, 1, 0, 0, DataFormat.NHWC).
        setName("depWiseBackprop")
    val input = T(Tensor[Int](T(4, 24, 24, 3)),
      Tensor[Float](2, 2, 3, 1).apply1(_ => Random.nextFloat()),
      Tensor[Float](4, 23, 23, 3).apply1(_ => Random.nextFloat()))
    runSerializationTest(depWiseBackprop, input)
  }

  "DepthwiseConv2D serializer" should "work properly" in {
    val depWIseConv2d = DepthwiseConv2D[Float](1, 1, 0, 0).setName("depWIseConv2d")
    val input = T(Tensor[Float](4, 24, 24, 3).apply1(_ => Random.nextFloat()),
      Tensor[Float](2, 2, 3, 1).apply1(_ => Random.nextFloat()))
    runSerializationTest(depWIseConv2d, input)
  }

  "DepthwiseConv2DBackpropFilter serializer" should "work properly" in {
    val depWiseConv2dBackProp = DepthwiseConv2DBackpropFilter[Float](1,
      1, 0, 0, DataFormat.NHWC).setName("depWiseConv2dBackProp")
    val input = T(Tensor[Float](4, 24, 24, 3).apply1(_ => Random.nextFloat()),
      Tensor[Int](T(2, 2, 3, 1)),
      Tensor[Float](4, 23, 23, 3).apply1(_ => Random.nextFloat()))
    runSerializationTest(depWiseConv2dBackProp, input)
  }

  "EluGrad serializer" should "work properly" in {
    val eluGrad = EluGrad[Float, Float]().setName("eluGrad")
    val inputTensor = Tensor[Float](5).apply1(_ => Random.nextFloat())
    val grad = Tensor[Float](5).apply1(_ => Random.nextFloat())
    val input = T(inputTensor, grad)
    runSerializationTest(eluGrad, input)
  }

  "Equal serializer" should "work properly" in {
    val equal = Equal[Float]().setName("equal")
    val input = T(Tensor[Float](5).apply1(_ => Random.nextFloat()),
      Tensor[Float](5).apply1(_ => Random.nextFloat()))
    runSerializationTest(equal, input,
      equal.asInstanceOf[ModuleToOperation[Float]].module.getClass)
  }

  "ExpOps serializer" should "work properly" in {
    val expOps = ExpOps[Float, Float]().setName("expOps")
    val input = Tensor[Float](5).apply1(_ => Random.nextFloat())
    runSerializationTest(expOps, input)
  }

  "Expm1 serializer" should "work properly" in {
    val expm1 = Expm1[Float, Float]().setName("expm1")
    val input = Tensor[Float](5).apply1(_ => Random.nextFloat())
    runSerializationTest(expm1, input)
  }

  "Floor serializer" should "work properly" in {
    val floor = Floor[Float]().setName("floor")
    val input = Tensor[Float](5).apply1(_ => Random.nextFloat())
    runSerializationTest(floor, input, floor.
      asInstanceOf[ModuleToOperation[Float]].module.getClass)
  }

  "FloorDiv serializer" should "work properly" in {
    val floorDiv = FloorDiv[Float, Float]().setName("floorDiv")
    val input1 = Tensor[Float](5).fill(1.0f)
    val input2 = Tensor[Float](5).fill(2.0f)
    val input = T(input1, input2)
    runSerializationTest(floorDiv, input)
  }

  "FloorMod serializer" should "work properly" in {
    val floorMod = FloorMod[Float, Float]().setName("floorMod")
    val input1 = Tensor[Float](5).fill(1.0f)
    val input2 = Tensor[Float](5).fill(2.0f)
    val input = T(input1, input2)
    runSerializationTest(floorMod, input)
  }

  "FusedBatchNorm serializer" should "work properly" in {
    val fusedBatchNorm = FusedBatchNorm[Float]().setName("fusedBatchNorm")
    val input = T(Tensor[Float](4, 8, 8, 256).apply1(_ => Random.nextFloat()),
      Tensor[Float](256).apply1(_ => Random.nextFloat()),
      Tensor[Float](256).apply1(_ => Random.nextFloat()),
      Tensor[Float](0),
      Tensor[Float](0))
    runSerializationTest(fusedBatchNorm, input)
  }

  "FusedBatchNormGrad serializer" should "work properly" in {
    val fbatchNormGrad = FusedBatchNormGrad[Float]().setName("fbatchNormGrad")
    val input = T(Tensor[Float](4, 8, 8, 256).rand(),
      Tensor[Float](4, 8, 8, 256).apply1(_ => Random.nextFloat()),
      Tensor[Float](256).apply1(_ => Random.nextFloat()),
      Tensor[Float](256).apply1(_ => Random.nextFloat()),
      Tensor[Float](256).apply1(_ => Random.nextFloat()))
    runSerializationTest(fbatchNormGrad, input)
  }

  "Greater serializer" should "work properly" in {
    val greater = Greater[Float]().setName("greater")
    val input1 = Tensor[Float](5).apply1(_ => Random.nextFloat())
    val input2 = Tensor[Float](5).apply1(_ => Random.nextFloat())
    val input = T(input1, input2)
    runSerializationTest(greater, input, greater.
      asInstanceOf[ModuleToOperation[Float]].module.getClass)
  }

  "GreaterEqual serializer" should "work properly" in {
    val greaterEqual = GreaterEqual[Float]().setName("greaterEqual")
    val input1 = Tensor[Float](5).apply1(_ => Random.nextFloat())
    val input2 = Tensor[Float](5).apply1(_ => Random.nextFloat())
    val input = T(input1, input2)
    runSerializationTest(greaterEqual, input, greaterEqual
      .asInstanceOf[ModuleToOperation[Float]].module.getClass)
  }

  "Indicator serializer" should "work properly" in {
    val indicatorCol = IndicatorCol[Float](
      feaLen = 4,
      isCount = true
    ).setName("indicatorCol")
    val input = Tensor.sparse(
      Array(Array(0, 1, 1, 2, 2, 3, 3, 3),
        Array(0, 0, 3, 0, 1, 0, 1, 2)),
      Array(3, 1, 2, 0, 3, 1, 2, 2),
      Array(4, 4)
    )
    runSerializationTest(indicatorCol, input)
  }

  "InTopK serializer" should "work properly" in {
    val inTopK = InTopK[Float](2).setName("inTopK")
    val input1 = Tensor[Float](2, 5).apply1(_ => Random.nextFloat())
    val input2 = Tensor[Int](2).fill(1)
    val input = T(input1, input2)
    runSerializationTest(inTopK, input)
  }

  "Inv serializer" should "work properly" in {
    val inv = Inv[Float, Float]().setName("inv")
    val input = Tensor[Float](2, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(inv, input)
  }

  "InvGrad serializer" should "work properly" in {
    val invGrad = InvGrad[Float, Float]().setName("invGrad")
    val input = T(Tensor[Float](2, 5).apply1(_ => Random.nextFloat()),
      Tensor[Float](2, 5).apply1(_ => Random.nextFloat()))
    runSerializationTest(invGrad, input)
  }

  "IsFinite serializer" should "work properly" in {
    val isFinite = IsFinite[Float, Float]().setName("isFinite")
    val input = Tensor[Float](2, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(isFinite, input)
  }

  "IsInf serializer" should "work properly" in {
    val isInf = IsInf[Float, Float]().setName("isInf")
    val input = Tensor[Float](2, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(isInf, input)
  }

  "IsNan serializer" should "work properly" in {
    val isNan = IsNan[Float, Float]().setName("isInf")
    val input = Tensor[Float](2, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(isNan, input)
  }

  "Kv2Tensor" should "work properly" in {
    val kv2tensor = Kv2Tensor[Float, Float](
      kvDelimiter = ",", itemDelimiter = ":", transType = 0
    ).setName("kv2tensor")
    val input = T(
      Tensor[String](
        T(T("0:0.1,1:0.2"), T("1:0.3,3:0.5"), T("2:0.15,4:0.25"))),
      Tensor[Int](Array(5), shape = Array[Int]())
    )
    runSerializationTest(kv2tensor, input)
  }

  "L2Loss serializer" should "work properly" in {
    val l2loss = L2Loss[Float]().setName("l2loss")
    val input = Tensor[Float](2, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(l2loss, input,
      l2loss.asInstanceOf[ModuleToOperation[Float]].module.getClass)
  }

  "Less serializer" should "work properly" in {
    val less = Less[Float]().setName("less")
    val input1 = Tensor[Float](5).apply1(_ => Random.nextFloat())
    val input2 = Tensor[Float](5).apply1(_ => Random.nextFloat())
    val input = T(input1, input2)
    runSerializationTest(less, input, less
      .asInstanceOf[ModuleToOperation[Float]].module.getClass)
  }

  "CategoricalColHashBucket" should "work properly" in {
    val categoricalColHashBucket = CategoricalColHashBucket[Float](
      hashBucketSize = 100
    ).setName("categoricalColHashBucket")
    val input = Tensor[String](T(T(1), T(2), T(3)))
    runSerializationTest(categoricalColHashBucket, input)
  }

  "CategoricalColVocaList" should "work properly" in {
    val categoricalColVocaList = CategoricalColVocaList[Float](
      vocaList = Array("A", "B", "C"),
      strDelimiter = ",",
      isSetDefault = false,
      numOovBuckets = 0
    ).setName("categoricalColVocaList")
    val input = Tensor[String](T(T("A"), T("B"), T("C"), T("D")))
    runSerializationTest(categoricalColVocaList, input)
  }

  "LessEqual serializer" should "work properly" in {
    val lessEqual = LessEqual[Float]().setName("lessEqual")
    val input1 = Tensor[Float](5).apply1(_ => Random.nextFloat())
    val input2 = Tensor[Float](5).apply1(_ => Random.nextFloat())
    val input = T(input1, input2)
    runSerializationTest(lessEqual, input, lessEqual
      .asInstanceOf[ModuleToOperation[Float]].module.getClass)
  }

  "LogicalAnd serializer" should "work properly" in {
    val logicalAnd = LogicalAnd[Float].setName("logicalAnd")
    val input = T(Tensor[Boolean](T(true, false)), Tensor[Boolean](T(true, false)))
    runSerializationTest(logicalAnd, input, logicalAnd.
      asInstanceOf[ModuleToOperation[Float]].module.getClass)
  }

  "LogicalNot serializer" should "work properly" in {
    val logicalNot = LogicalNot[Float].setName("logicalNot")
    val input = Tensor[Boolean](T(true, false))
    runSerializationTest(logicalNot, input, logicalNot
      .asInstanceOf[ModuleToOperation[Float]].module.getClass)
  }

  "LogicalOr serializer" should "work properly" in {
    val logicalOr = LogicalOr[Float].setName("logicalOr")
    val input = T(Tensor[Boolean](T(true, false)), Tensor[Boolean](T(true, false)))
    runSerializationTest(logicalOr, input, logicalOr
      .asInstanceOf[ModuleToOperation[Float]].module.getClass)
  }

  "LRNGrad serializer" should "work properly" in {
    val lrnGrad = LRNGrad[Float]().setName("lrnGrad")
    val input = T(Tensor[Float](4, 8, 8, 3).apply1(_ => Random.nextFloat()),
      Tensor[Float](4, 8, 8, 3).apply1(_ => Random.nextFloat()),
      Tensor[Float](4, 8, 8, 3).apply1(_ => Random.nextFloat())
    )
    runSerializationTest(lrnGrad, input)
  }

  "Maximum serializer" should "work properly" in {
    val maxiMum = Maximum[Float, Float]().setName("maxiMum")
    val input = T(Tensor[Float](5).apply1(_ => Random.nextFloat()),
      Tensor[Float](5).apply1(_ => Random.nextFloat()))
    runSerializationTest(maxiMum, input)
  }

  "MaxPool serializer" should "work properly" in {
    val maxPool = MaxPool[Float](
      Array(1, 2, 3, 1),
      Array(1, 2, 1, 1),
      "VALID").setName("maxPool")
    val input = Tensor[Float](1, 4, 3, 3).apply1(_ => Random.nextFloat())
    runSerializationTest(maxPool, input, maxPool.
      asInstanceOf[ModuleToOperation[Float]].module.getClass)

  }

  "MaxPoolGrad serializer" should "work properly" in {
    val maxPoolGrad = MaxPoolGrad[Float](2, 1, 1, 1, 0, 0, DataFormat.NCHW).
      setName("maxPoolGrad")
    val input = T(Tensor[Float](1, 3, 3).apply1(_ => Random.nextFloat()),
      Tensor[Float](),
      Tensor[Float](1, 1, 1).apply1(_ => Random.nextFloat()))
    runSerializationTest(maxPoolGrad, input)
  }

  "Mimimum serializer" should "work properly" in {
    val minimum = Minimum[Float, Float]().setName("minimum")
    val input = T(Tensor[Float](5).apply1(_ => Random.nextFloat()),
      Tensor[Float](5).apply1(_ => Random.nextFloat()))
    runSerializationTest(minimum, input)
  }

  "Mod serializer" should "work properly" in {
    val mod = Mod[Float, Float]().setName("mod")
    val input1 = Tensor[Float](5).fill(1.0f)
    val input2 = Tensor[Float](5).fill(2.0f)
    val input = T(input1, input2)
    runSerializationTest(mod, input)
  }

  "ModuleToOperation serializer" should "work properly" in {
    val moduleToOperation = ModuleToOperation[Float](new LogicalOr()).
      setName("moduleToOperation")
    val input = T(Tensor[Boolean](T(true, false)), Tensor[Boolean](T(true, false)))
    runSerializationTest(moduleToOperation, input)
  }


  "TensorModuleWrapper serializer" should "work properly" in {
    val tensorModuleWrapper = TensorModuleWrapper[Float, Float](BigDLSoftPlus[Float]()).
      setName("moduleToOperation")
    val input = Tensor[Float](T(1.0f, 1.0))
    runSerializationTest(tensorModuleWrapper, input)
  }

  "NoOp serializer" should "work properly" in {
    val noOp = new com.intel.analytics.bigdl.nn.tf.NoOp[Float]().setName("noOp")
    val input = Tensor[Float](5).apply1(_ => Random.nextFloat())
    runSerializationTest(noOp, input)
  }

  "NotEqual serializer" should "work properly" in {
    val notEqual = NotEqual[Float].setName("notEqual")
    val input = T(Tensor[Boolean](T(true, false)), Tensor[Boolean](T(true, false)))
    runSerializationTest(notEqual, input, notEqual
      .asInstanceOf[ModuleToOperation[Float]].module.getClass)
  }

  "OneHot serializer" should "work properly" in {
    val oneHot = OneHot[Float, Float](axis = -1).setName("oneHot")
    val input =
      T(Tensor[Long](T(0, 2, -1, 1)),
        Tensor[Int](Array(3), shape = Array[Int]()),
        Tensor[Float](Array(0.5f), shape = Array[Int]()),
        Tensor[Float](Array(0.0f), shape = Array[Int]()))
    runSerializationTest(oneHot, input, oneHot
      .asInstanceOf[ModuleToOperation[Float]].module.getClass)
  }

  "Pad serializer" should "work properly" in {
    val pad = Pad[Float, Float](mode = "CONSTANT", 0.0f).setName("pad")
    val inputTensor = Tensor[Float](2, 2, 3).apply1(_ => Random.nextFloat())
    val padding = Tensor[Int](T(T(1, 2), T(1, 2), T(1, 2)))
    val input = T(inputTensor, padding)
    runSerializationTest(pad, input, pad.
      asInstanceOf[ModuleToOperation[Float]].module.getClass)
  }

  "ParseExample serializer" should "work properly" in {
    import com.intel.analytics.bigdl.utils.tf.TFTensorNumeric.NumericByteString

    val floatBuilder = FloatList.newBuilder()
      .addValue(0.0f).addValue(1.0f).addValue(2.0f)
    val floatFeature = Feature.newBuilder().setFloatList(floatBuilder).build()

    val longBuilder = Int64List.newBuilder()
      .addValue(0).addValue(1).addValue(2)
    val longFeature = Feature.newBuilder().setInt64List(longBuilder).build()

    val bytesBuilder = BytesList.newBuilder().addValue(ByteString.copyFromUtf8("abcd"))
    val bytesFeature = Feature.newBuilder().setBytesList(bytesBuilder).build()

    val features = Features.newBuilder()
      .putFeature("floatFeature", floatFeature)
      .putFeature("longFeature", longFeature)
      .putFeature("bytesFeature", bytesFeature)
    val example = Example.newBuilder().setFeatures(features).build()
    val length = example.getSerializedSize
    val data = new Array[Byte](length)
    val outputStream = CodedOutputStream.newInstance(data)
    example.writeTo(outputStream)

    val exampleParser = ParseExample[Float](3, Seq(FloatType, LongType, StringType),
      Seq(Array(3), Array(3), Array())).setName("parseExample")

    val serialized = Tensor[ByteString](Array(ByteString.copyFrom(data)), Array[Int](1))
    val names = Tensor[ByteString]()
    val key1 = Tensor[ByteString](Array(ByteString.copyFromUtf8("floatFeature")), Array[Int]())
    val key2 = Tensor[ByteString](Array(ByteString.copyFromUtf8("longFeature")), Array[Int]())
    val key3 = Tensor[ByteString](Array(ByteString.copyFromUtf8("bytesFeature")), Array[Int]())

    val default1 = Tensor[Float]()
    val default2 = Tensor[Long]()
    val default3 = Tensor[ByteString]()
    val input = T(serialized, names, key1, key2, key3, default1, default2, default3)
    runSerializationTest(exampleParser, input)
  }

  "PowOps serializer" should "work properly" in {
    val pow = PowOps[Float]().setName("powOps")
    val v = Tensor[Float](T(2))
    val t = Tensor[Float](T(1, 2, 3))
    val input = (T(t, v))
    runSerializationTest(pow, input)
  }

  "Prod serializer" should "work properly" in {
    val prod = Prod[Float](-1, false).setName("prod")
    val input = Tensor[Float](3, 3).apply1(_ => Random.nextFloat())
    runSerializationTest(prod, input, prod.
      asInstanceOf[ModuleToOperation[Float]].module.getClass)
  }

  "RandomUniform serializer" should "work properly" in {
    val randomUniform = RandomUniform[Float, Float](10, 20).
      setName("randomUniform")
    val input = Tensor[Int](T(1, 2, 3))
    runSerializationTest(randomUniform, input, randomUniform.
      asInstanceOf[ModuleToOperation[Float]].module.getClass)
  }

  "RangeOps serializer" should "work properly" in {
    val rangeOps = RangeOps[Float, Float]().setName("rangeOps")
    val input = T(Tensor[Float](T(1)), Tensor[Float](T(10)), Tensor[Float](T(1)))
    runSerializationTest(rangeOps, input)
  }

  "Rank serializer" should "work properly" in {
    val rank = Rank[Float].setName("rank")
    val input = Tensor[Float](3, 3).apply1(_ => Random.nextFloat())
    runSerializationTest(rank, input, rank.
      asInstanceOf[ModuleToOperation[Float]].module.getClass)
  }

  "Relu6Grad serializer" should "work properly" in {
    val relu6Grad = Relu6Grad[Float, Float]().setName("relu6Grad")
    val input = T(Tensor[Float](5).apply1(_ => Random.nextFloat()),
      Tensor[Float](5).apply1(_ => Random.nextFloat()))
    runSerializationTest(relu6Grad, input)
  }

  "ReluGrad serializer" should "work properly" in {
    val reluGrad = ReluGrad[Float]
    val input = T(Tensor[Float](2, 2, 2).apply1(_ => Random.nextFloat()),
      Tensor[Float](2, 2, 2).apply1(_ => Random.nextFloat()))
    runSerializationTest(reluGrad, input)
  }

  "ResizeBilinearOps serializer" should "work properly" in {
    val resizeBilinearOps = ResizeBilinearOps[Float](false).
      setName("resizeBiLinearOps")
    val input = T(Tensor[Float](1, 3, 2, 3).apply1(_ => Random.nextFloat()),
      Tensor[Int](T(3, 2)))
    runSerializationTest(resizeBilinearOps, input)
  }

  "Rint serializer" should "work properly" in {
    val rint = Rint[Float]().setName("rint")
    val input = Tensor[Float](3, 3).apply1(_ => Random.nextFloat())
    runSerializationTest(rint, input)
  }

  "Round serializer" should "work properly" in {
    val round = Round[Float, Float]().setName("round")
    val input = Tensor[Float](3, 3).apply1(_ => Random.nextFloat())
    runSerializationTest(round, input)
  }

  "RsqrtGrad serializer" should "work properly" in {
    val rsqrtGrad = RsqrtGrad[Float, Float].setName("rsqrtGrad")
    val input = T(Tensor[Float](3, 3).apply1(_ => Random.nextFloat()),
      Tensor[Float](3, 3).apply1(_ => Random.nextFloat()))
    runSerializationTest(rsqrtGrad, input)
  }

  "SegmentSum serializer" should "work properly" in {
    val sgSum = SegmentSum[Float].setName("segmentSum")
    val input = T(Tensor[Float](10, 3).apply1(_ => Random.nextFloat()),
      Tensor[Int](T(0, 0, 0, 1, 2, 3, 3, 4, 4, 4)))
    runSerializationTest(sgSum, input)
  }

  "SelectOps serializer" should "work properly" in {
    val select = SelectOps[Float]().setName("select")
    val cond = Tensor.scalar[Boolean](true)
    val t = Tensor[Int](T(1))
    val e = Tensor[Int](T(2))
    val input = T(cond, t, e)
    runSerializationTest(select, input)
  }

  "SigmoidGrad serializer" should "work properly" in {
    val sigMoidGrad = SigmoidGrad[Float, Float]().setName("sigMoidGrad")
    val input = T(Tensor[Float](2, 2, 2).apply1(_ => Random.nextFloat()),
      Tensor[Float](2, 2, 2).apply1(_ => Random.nextFloat()))
    runSerializationTest(sigMoidGrad, input)
  }

  "Sign serializer" should "work properly" in {
    val sign = Sign[Float, Float]().setName("sign")
    val input = Tensor[Float](3, 3).apply1(_ => Random.nextFloat())
    runSerializationTest(sign, input)
  }

  "SelectTensor serializer" should "work properly" in {
    val transformer = (TensorOp[Float]() ** 3 * 4.5f).ceil
    val select = SelectTensor(Tensor.scalar("2"), transformer)
    val t1 = Tensor[Float](3, 4).randn()
    val t2 = Tensor[Float](2, 3).randn()
    val input = T().update(Tensor.scalar(1), t1).update(Tensor.scalar("2"), t2)
    runSerializationTest(select, input)
  }

  "Slice serializer" should "work properly" in {
    val slice = Slice[Float](begin = Array(0, 1, 1),
      size = Array(2, -1, 1)).setName("slice")
    val input = Tensor[Float](3, 2, 3).apply1(_ => Random.nextFloat())
    runSerializationTest(slice, input, slice.
      asInstanceOf[ModuleToOperation[Float]].module.getClass)
  }

  "SoftplusGrad serializer" should "work properly" in {
    val sofplusGrad = SoftplusGrad[Float, Float].setName("sofplusGrad")
    val input = T(Tensor[Float](2, 2, 2).apply1(_ => Random.nextFloat()),
      Tensor[Float](2, 2, 2).apply1(_ => Random.nextFloat()))
    runSerializationTest(sofplusGrad, input)
  }

  "SoftSignGrad serializer" should "work properly" in {
    val softSign = SoftsignGrad[Float, Float].setName("softSign")
    val input = T(Tensor[Float](2, 2, 2).apply1(_ => Random.nextFloat()),
      Tensor[Float](2, 2, 2).apply1(_ => Random.nextFloat()))
    runSerializationTest(softSign, input)
  }

  "SqrtGrad serializer" should "work properly" in {
    val sqrtGrad = SqrtGrad[Float, Float].setName("sqrtGrad")
    val input = T(Tensor[Float](2, 2, 2).apply1(_ => Random.nextFloat()),
      Tensor[Float](2, 2, 2).apply1(_ => Random.nextFloat()))
    runSerializationTest(sqrtGrad, input)
  }

  "SquaredDifference serializer" should "work properly" in {
    val squareDiff = SquaredDifference[Float]().setName("squareDiff")
    val input = T(Tensor[Float](2, 2, 2).apply1(_ => Random.nextFloat()),
      Tensor[Float](2, 2, 2).apply1(_ => Random.nextFloat()))
    runSerializationTest(squareDiff, input)
  }

  "Substr serializer" should "work properly" in {
    import com.intel.analytics.bigdl.utils.tf.TFTensorNumeric.NumericByteString
    val subStr = Substr[Float]().setName("subStr")
    val input = T(Tensor.scalar[ByteString](ByteString.copyFromUtf8("HelloBigDL")),
      Tensor.scalar[Int](0), Tensor.scalar[Int](5))
    runSerializationTest(subStr, input)
  }

  "SumOps serializer" should "work properly" in {
    val sumOps = SumOps[Float, Float]().setName("sumOps")
    val input = T(Tensor[Float](2, 2).apply1(_ => Random.nextFloat()),
      Tensor[Float]())
    runSerializationTest(sumOps, input)
  }

  "TileOps serializer" should "work properly" in {
    val tileOps = TileOps[Float]().setName("tileOps")
    val input = T(Tensor[Float](2, 3, 3).apply1(_ => Random.nextFloat()),
      Tensor[Int](T(2, 1, 2)))
    runSerializationTest(tileOps, input, tileOps.
      asInstanceOf[ModuleToOperation[Float]].module.getClass)
  }

  "TensorOp serializer" should "work properly" in {
    val op = (((TensorOp[Float]() + 1.5f) ** 2) -> TensorOp.sigmoid()
      ).setName("TensorOP")
    val input = Tensor[Float](3, 3).apply1(_ => Random.nextFloat())
    runSerializationTest(op, input)
  }

  "TopK serializer" should "work properly" in {
    val topk = TopK[Float, Float](2).setName("topK")
    val input = Tensor[Float](3, 3).apply1(_ => Random.nextFloat())
    runSerializationTest(topk, input)
  }

  "TruncateDiv serializer" should "work properly" in {
    val truncateDiv = TruncateDiv[Float, Float]().setName("truncateDiv")
    val input = T(Tensor[Float](5).fill(1.0f), Tensor[Float](5).fill(2.0f))
    runSerializationTest(truncateDiv, input)
  }

  "TruncatedNormal serializer" should "work properly" in {
    val truncateNormal = TruncatedNormal[Float, Float](10, 20).setName("truncateNormal")
    val input = Tensor[Int](T(1, 2, 3))
    runSerializationTest(truncateNormal, input, truncateNormal.
      asInstanceOf[ModuleToOperation[Float]].module.getClass)
  }

  "BiasAdd serializer" should "work properly" in {
    val biasAdd = BiasAdd[Float]().setName("biasAdd")
    val input = T(Tensor[Float](2, 3, 3).apply1(_ => Random.nextFloat()),
      Tensor[Float](3).apply1(_ => Random.nextFloat()))
    runSerializationTest(biasAdd, input)
  }
  "Const serializer" should "work properly" in {
    val value = Tensor[Float](3).apply1(_ => Random.nextFloat())
    val const = Const[Float, Float](value).setName("const")
    val input = Tensor[Float](3).apply1(_ => Random.nextFloat())
    runSerializationTest(const, input)
  }

  "Fill serializer" should "work properly" in {
    val fill = Fill[Float]().setName("fill")
    val shape = Tensor[Int](T(2, 3))
    val value = Tensor[Float](Array(0.1f), Array[Int]())
    val input = T(shape, value)
    runSerializationTest(fill, input)
  }

  "Log1p serializer" should "work properly" in {
    val log1p = Log1p[Float, Float]().setName("log1p")
    val input = Tensor[Float](3).apply1(_ => Random.nextFloat())
    runSerializationTest(log1p, input)
  }

  "Shape serializer" should "work properly" in {
    val shape = Shape[Float]().setName("shape")
    val input = Tensor[Float](3).apply1(_ => Random.nextFloat())
    runSerializationTest(shape, input)
  }
  "MeanLoadTF serializer" should "work properly" in {
    val meanLoadTF = new MeanLoadTF[Float]("Float", false).setName("meanLoadTF")
    val input = T(Tensor[Float](1, 2).apply1(_ => Random.nextFloat()),
      Tensor[Int](T(1, 1)))
    runSerializationTest(meanLoadTF, input)
  }

  "ConcatV2LoadTF serializer" should "work properly" in {
    val concatv2 = new ConcatV2LoadTF[Float]().setName("concatv2LoadTF")
    val input = T(Tensor[Float](1, 2).apply1(_ => Random.nextFloat()),
      Tensor[Float](1, 2).apply1(_ => Random.nextFloat()),
      Tensor[Int](T(1)))
    runSerializationTest(concatv2, input)
  }

  "ExpandDimsLoadTF serializer" should "work properly" in {
    val expandDim = new ExpandDimsLoadTF[Float]().setName("expandDim")
    val input = T(Tensor[Float](1, 2).apply1(_ => Random.nextFloat()),
      Tensor.scalar[Int](1))
    runSerializationTest(expandDim, input)
  }

  "PadLoadTF serializer" should "work properly" in {
    val padLoadTF = new PadLoadTF[Float]().setName("PadLoadTF")
    val input = T(Tensor[Float](5, 5, 5).apply1(_ => Random.nextFloat()),
      Tensor[Int](T(T(1, 1), T(1, 1))))
    runSerializationTest(padLoadTF, input)
  }

  "ProdLoadTF serializer" should "work properly" in {
    val prodLoadTF = new ProdLoadTF[Float]().setName("prodLoadTF")
    val input = T(Tensor[Float](5, 5, 5).apply1(_ => Random.nextFloat()),
      Tensor.scalar[Int](1))
    runSerializationTest(prodLoadTF, input)
  }

  "ReshapeLoadTF serializer" should "work properly" in {
    val reshapeLoadTF = new ReshapeLoadTF[Float]().setName("reshapeLoadTF")
    val input = T(Tensor[Float](5, 5, 5).apply1(_ => Random.nextFloat()),
      Tensor[Int](T(1, 5, 25)))
    runSerializationTest(reshapeLoadTF, input)
  }

  "SliceLoadTF serializer" should "work properly" in {
    val sliceLoadTF = new SliceLoadTF[Float]().setName("sliceLoadTF")
    val input = T(Tensor[Float](3, 2, 3).apply1(_ => Random.nextFloat()),
      Tensor[Int](T(0, 1, 1)),
      Tensor[Int](T(2, -1, 1)))
    runSerializationTest(sliceLoadTF, input)
  }

  "StridedSliceLoadTF serializer" should "work properly" in {
    val strideSliceLoadTF = new StridedSliceLoadTF[Float, Float]().
      setName("strideSliceLoadTF")
    val input = T(Tensor[Float](2, 2, 2).apply1(_ => Random.nextFloat()),
      Tensor[Int](T(0)),
      Tensor[Int](T(1)),
      Tensor[Int](T(1))
    )
    runSerializationTest(strideSliceLoadTF, input)
  }

  "SplitLoadTF serializer" should "work properly" in {
    val splitLoadTF = new SplitLoadTF[Float](1).setName("splitLoadTD")
    val input = T(Tensor[Int](T(1)),
      Tensor[Float](1, 6, 2).apply1(_ => Random.nextFloat())
    )
    runSerializationTest(splitLoadTF, input)
  }

  "TransposeLoadTF serializer" should "work properly" in {
    val transposeLoadTF = new TransposeLoadTF[Float]().setName("transposeLoadTF")
    val input = T(Tensor[Float](1, 6, 2).apply1(_ => Random.nextFloat()),
      Tensor[Int](T(0, 1))
    )
    runSerializationTest(transposeLoadTF, input)
  }

  "TopKV2LoadTF serializer" should "work properly" in {
    val topkv2LoadTF = new TopKV2LoadTF[Float](false, "Float").
      setName("topkv2LoadTF")
    val input = T(Tensor[Float](3, 3).apply1(_ => Random.nextFloat()),
      Tensor.scalar[Int](2)
    )
    runSerializationTest(topkv2LoadTF, input)
  }

  "Digamma serializer" should "work properly" in {
    val module = Digamma[Float, Float]()

    val input = Tensor[Float](1, 5, 3, 4).rand()
    runSerializationTest(module, input)
  }

  "Lgamma serializer" should "work properly" in {
    val module = Lgamma[Float, Float]()

    val input = Tensor[Float](1, 5, 3, 4).rand()
    runSerializationTest(module, input)
  }

  "Erf serializer" should "work properly" in {
    val module = Erf[Float, Float]()

    val input = Tensor[Float](1, 5, 3, 4).rand()
    runSerializationTest(module, input)
  }

  "Erfc serializer" should "work properly" in {
    val module = Erfc[Float, Float]()

    val input = Tensor[Float](1, 5, 3, 4).rand()
    runSerializationTest(module, input)
  }

  "TanhGrad serializer" should "work properly" in {
    val module = TanhGrad[Float, Float]()

    val input = T(Tensor[Float](1, 5, 3, 4).rand(), Tensor[Float](1, 5, 3, 4).rand())

    runSerializationTest(module, input)
  }

  "Dilation2D serializer" should "work properly" in {
    val module = Dilation2D[Float, Float](
      Array(1, 3, 2, 1), Array(1, 2, 3, 1), "same")

    val input = T(Tensor[Float](4, 32, 32, 3).rand(), Tensor[Float](3, 4, 3).rand())

    runSerializationTest(module, input)
  }

  "Dilation2DBackpropFilter serializer" should "work properly" in {
    val module = Dilation2DBackpropFilter[Float, Float](
      Array(1, 3, 2, 1), Array(1, 2, 3, 1), "same")

    val input = T(Tensor[Float](4, 32, 32, 3).rand(),
      Tensor[Float](3, 4, 3).rand(),
      Tensor[Float](4, 11, 16, 3).rand())

    runSerializationTest(module, input)
  }

  "Dilation2DBackpropInput serializer" should "work properly" in {
    val module = Dilation2DBackpropInput[Float, Float](
      Array(1, 3, 2, 1), Array(1, 2, 3, 1), "same")

    val input = T(Tensor[Float](4, 32, 32, 3).rand(),
      Tensor[Float](3, 4, 3).rand(),
      Tensor[Float](4, 11, 16, 3).rand())

    runSerializationTest(module, input)
  }

  "Conv3D serializer" should "work properly" in {
    val module = Conv3D[Float](1, 2, 3, 0, 0, 0, DataFormat.NHWC)
    val input = Tensor[Float](4, 20, 30, 40, 3).rand()
    val filter = Tensor[Float](2, 3, 4, 3, 4).rand()
    runSerializationTest(module, T(input, filter))
  }

  "Conv3DBackpropFilter serializer" should "work properly" in {
    val module = Conv3DBackpropFilter[Float](1, 2, 3, 0, 0, 0, DataFormat.NHWC)
    val input = Tensor[Float](4, 20, 30, 40, 3).rand()
    val filter = Tensor[Float](2, 3, 4, 3, 4).rand()
    val outputBackprop = Tensor[Float](4, 19, 14, 13, 4)

    runSerializationTest(module, T(input, filter, outputBackprop))
  }

  "Conv3DBackpropInput serializer" should "work properly" in {
    val module = Conv3DBackpropInput[Float](1, 2, 3, 0, 0, 0, DataFormat.NHWC)
    val input = Tensor[Float](4, 20, 30, 40, 3).rand()
    val filter = Tensor[Float](2, 3, 4, 3, 4).rand()
    val outputBackprop = Tensor[Float](4, 19, 14, 13, 4).rand()

    runSerializationTest(module, T(input, filter, outputBackprop))
  }

  "Conv3DBackpropFilterV2 serializer" should "work properly" in {
    val module = Conv3DBackpropFilterV2[Float](1, 2, 3, 0, 0, 0, DataFormat.NHWC)
    val input = Tensor[Float](4, 20, 30, 40, 3).rand()
    val filter = Tensor[Int](Array(2, 3, 4, 3, 4), Array(5))
    val outputBackprop = Tensor[Float](4, 19, 14, 13, 4).rand()

    runSerializationTest(module, T(input, filter, outputBackprop))
  }

  "Conv3DBackpropInputV2 serializer" should "work properly" in {
    val module = Conv3DBackpropInputV2[Float](1, 2, 3, 0, 0, 0, DataFormat.NHWC)
    val inputSize = Tensor[Int](Array(4, 20, 30, 40, 3), Array(5))
    val filter = Tensor[Float](2, 3, 4, 3, 4).rand()
    val outputBackprop = Tensor[Float](4, 19, 14, 13, 4).rand()

    runSerializationTest(module, T(inputSize, filter, outputBackprop))
  }

  "ResizeBilinearGrad serializer" should "work properly" in {
    val module = ResizeBilinearGrad[Float](true)
    val input = T(Tensor[Float](1, 224, 224, 3).rand(),
      Tensor[Float](1, 64, 64, 3).rand())
    val outputBackprop = Tensor[Float](4, 19, 14, 13, 4).rand()

    runSerializationTest(module, input)
  }

  "Control Ops serializer" should "work properly" in {
    val input = Input[Float]("input")

    val conditionInput = Input[Float]("conditionInput")
    val const = new com.intel.analytics.bigdl.nn.tf.Const[Float, Float](Tensor(T(9))).inputs()
    val constEnter = new com.intel.analytics.bigdl.nn.tf.Enter[Float]("test_frame").inputs(const)
    val less = Less[Float]().inputs(constEnter, conditionInput)

    val updateInput = Input[Float]()
    val add = AddConstant[Float](1).inputs(updateInput)
    val addEnter = new com.intel.analytics.bigdl.nn.tf.Enter[Float]("test_frame").inputs(add)
    val echo = Echo[Float]().inputs(addEnter)

    val exit = ControlNodes.whileLoop[Float](
      (Seq(conditionInput), less),
      (Seq((updateInput, echo))),
      Seq(input),
      "while"
    )
    val model = Graph.dynamic[Float](Array(input), Array(exit(0)), None, false)
    runSerializationTestWithMultiClass(model, Tensor.scalar[Float](1), Array(
      addEnter.element.getClass.asInstanceOf[Class[_]],
      new com.intel.analytics.bigdl.nn.tf.NextIteration[Float, Float]().getClass,
      new com.intel.analytics.bigdl.nn.tf.Exit[Float]().getClass,
      new LoopCondition[Float]().getClass
    ))
  }

  "Stack operations serializer" should "work properly" in {
    import com.intel.analytics.bigdl.nn.ops._
    val data = Const[Float, Float](Tensor.scalar[Float](1)).inputs()
    val stack = new StackCreator[Float, Float]().inputs()
    val push = new com.intel.analytics.bigdl.nn.tf.StackPush[Float, Float]().inputs(stack, data)
    val ctr = new com.intel.analytics.bigdl.nn.tf.ControlDependency[Float]().inputs(push)
    val pop = new com.intel.analytics.bigdl.nn.tf.StackPop[Float, Float]().inputs(stack, ctr)
    val model = Graph.dynamic[Float](Array(stack), Array(pop))

    runSerializationTestWithMultiClass(model, Tensor.scalar(1), Array(
      stack.element.getClass.asInstanceOf[Class[_]],
      push.element.getClass.asInstanceOf[Class[_]],
      pop.element.getClass.asInstanceOf[Class[_]]
    ))
  }

  "TensorArray serializer R/W" should "work properly" in {
    import com.intel.analytics.bigdl.nn.ops._
    val tensorArray = new TensorArrayCreator[Float, Float]().inputs()
    val data = Const[Float, Float](Tensor.scalar[Float](1)).inputs()
    val index = Const[Float, Int](Tensor.scalar[Int](0)).inputs()
    val write = new TensorArrayWrite[Float, Float]().inputs((tensorArray, 1), (index, 1), (data, 1))
    val ctr = new com.intel.analytics.bigdl.nn.tf.ControlDependency[Float]().inputs(write)
    val read = new TensorArrayRead[Float, Float]().inputs((tensorArray, 1), (index, 1), (ctr, 1))
    val grad = new TensorArrayGrad[Float]("grad").inputs(tensorArray)
    val output = Identity[Float]().inputs((grad, 2))
    val model = Graph.dynamic[Float](Array(tensorArray), Array(read, output))

    runSerializationTestWithMultiClass(model, Tensor.scalar[Int](1), Array(
      tensorArray.element.getClass.asInstanceOf[Class[_]],
      write.element.getClass.asInstanceOf[Class[_]],
      read.element.getClass.asInstanceOf[Class[_]],
      grad.element.getClass.asInstanceOf[Class[_]]
    ))
  }

  "TensorArray serializer Gather/Scatter" should "work properly" in {
    import com.intel.analytics.bigdl.nn.ops._
    val tensorArray = new TensorArrayCreator[Float, Float]().inputs()
    val data = Const[Float, Float](Tensor[Float](3, 4).rand()).inputs()
    val indices = Const[Float, Int](Tensor[Int](T(0, 1, 2))).inputs()
    val scatter = new TensorArrayScatter[Float, Float]().inputs((tensorArray, 1), (indices, 1),
      (data, 1))
    val ctr = new com.intel.analytics.bigdl.nn.tf.ControlDependency[Float]().inputs(scatter)
    val gather = new TensorArrayGather[Float, Float]().inputs((tensorArray, 1), (indices, 1),
      (ctr, 1))
    val ctr2 = new com.intel.analytics.bigdl.nn.tf.ControlDependency[Float]().inputs(gather)
    val close = new TensorArrayClose[Float]().inputs((tensorArray, 1), (ctr2, 1))
    val model = Graph.dynamic[Float](Array(tensorArray), Array(gather, close))

    runSerializationTestWithMultiClass(model, Tensor.scalar[Int](10), Array(
      tensorArray.element.getClass.asInstanceOf[Class[_]],
      scatter.element.getClass.asInstanceOf[Class[_]],
      gather.element.getClass.asInstanceOf[Class[_]],
      close.element.getClass.asInstanceOf[Class[_]]
    ))
  }

  "TensorArray serializer Split/Concat" should "work properly" in {
    import com.intel.analytics.bigdl.nn.ops._
    val tensorArray = new TensorArrayCreator[Float, Float]().inputs()
    val data = Const[Float, Float](Tensor[Float](3, 4).rand()).inputs()
    val lengths = Const[Float, Int](Tensor[Int](T(1, 2))).inputs()
    val splitter = new TensorArraySplit[Float, Float]().inputs((tensorArray, 1), (data, 1),
      (lengths, 1))
    val ctr = new com.intel.analytics.bigdl.nn.tf.ControlDependency[Float]().inputs(splitter)
    val concat = new TensorArrayConcat[Float, Float]().inputs(tensorArray, ctr)
    val size = new TensorArraySize[Float]().inputs(tensorArray, ctr)
    val ctr2 = new com.intel.analytics.bigdl.nn.tf.ControlDependency[Float]().inputs(concat, size)
    val close = new TensorArrayClose[Float]().inputs((tensorArray, 1), (ctr2, 1))
    val model = Graph.dynamic[Float](Array(tensorArray), Array(concat, close, size))

    runSerializationTestWithMultiClass(model, Tensor.scalar[Int](2), Array(
      tensorArray.element.getClass.asInstanceOf[Class[_]],
      splitter.element.getClass.asInstanceOf[Class[_]],
      concat.element.getClass.asInstanceOf[Class[_]],
      close.element.getClass.asInstanceOf[Class[_]],
      size.element.getClass.asInstanceOf[Class[_]]
    ))
  }

  "ConcatOffset serializer" should "work properly" in {
    val module = new com.intel.analytics.bigdl.nn.tf.ConcatOffset[Float]()
    runSerializationTest(module, T(Tensor.scalar[Int](1), Tensor[Int](T(2, 2, 5, 7)),
      Tensor[Int](T(2, 3, 5, 7)), Tensor[Int](T(2, 4, 5, 7))))
  }

  "InvertPermutation serializer" should "work properly" in {
    val module = new com.intel.analytics.bigdl.nn.tf.InvertPermutation[Float]()
    runSerializationTest(module, Tensor[Int](T(0, 1, 2, 3, 4)))
  }

}
