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

import java.io.File
import java.io.{File => JFile}
import java.lang.reflect.Modifier

import com.google.protobuf.{ByteString, CodedOutputStream}
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat.NHWC

import scala.collection.JavaConverters._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, DataFormat}
import com.intel.analytics.bigdl.nn.ops.{All, Any, ApproximateEqual, ArgMax, Assert, Assign, AssignGrad, AvgPoolGrad, BatchMatMul, BiasAddGrad, BroadcastGradientArgs, Cast, Ceil, ControlNodes, Conv2D, Conv2DBackFilter, Conv2DTranspose, Conv3D, Conv3DBackpropFilter, Conv3DBackpropFilterV2, Conv3DBackpropInput, Conv3DBackpropInputV2, CrossEntropy, DecodeImage, DepthwiseConv2D, DepthwiseConv2DBackpropFilter, DepthwiseConv2DBackpropInput, Digamma, Dilation2D, Dilation2DBackpropFilter, Dilation2DBackpropInput, EluGrad, Equal, Erf, Erfc, Expm1, Floor, FloorDiv, FloorMod, FusedBatchNorm, FusedBatchNormGrad, Greater, GreaterEqual, InTopK, Inv, InvGrad, IsFinite, IsInf, IsNan, L2Loss, LRNGrad, Less, LessEqual, Lgamma, LogicalAnd, LogicalNot, LogicalOr, MaxPool, MaxPoolGrad, Maximum, MergeOps, Minimum, Mod, ModuleToOperation, NoOp, NotEqual, OneHot, Pad, ParseExample, Prod, RandomUniform, RangeOps, Rank, Relu6Grad, ReluGrad, ResizeBilinearGrad, ResizeBilinearOps, Rint, Round, RsqrtGrad, SegmentSum, SigmoidGrad, Sign, Slice, SoftplusGrad, SoftsignGrad, SqrtGrad, SquaredDifference, Substr, SwitchOps, TanhGrad, TopK, TruncateDiv, TruncatedNormal, Add => AddOps, DecodeGif => DecodeGifOps, DecodeJpeg => DecodeJpegOps, DecodePng => DecodePngOps, DecodeRaw => DecodeRawOps, Exp => ExpOps, Pow => PowOps, Select => SelectOps, Sum => SumOps, Tile => TileOps}
import com.intel.analytics.bigdl.nn.tf._
import com.intel.analytics.bigdl.nn.{DenseToSparse, SpatialDropout1D, _}
import com.intel.analytics.bigdl.optim.L2Regularizer
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.bigdl.utils.tf.TFRecordIterator
import com.intel.analytics.bigdl.utils.tf.loaders.{Pack => _, _}
import com.intel.analytics.bigdl.utils.{T, Table}
import org.reflections.Reflections
import org.reflections.scanners.SubTypesScanner
import org.reflections.util.{ClasspathHelper, ConfigurationBuilder, FilterBuilder}
import org.scalatest.{BeforeAndAfterAll, FlatSpec, Matchers}
import org.tensorflow.example._
import org.tensorflow.framework.DataType

import scala.collection.mutable
import scala.util.Random


class ModuleSerializerSpec extends FlatSpec with Matchers with BeforeAndAfterAll {

  val pkg = "com.intel.analytics.bigdl.nn"
  val postFix = "bigdl"
  val excluded = new mutable.HashSet[String]()
  val expected = new mutable.HashSet[String]()
  val tested = new mutable.HashSet[String]()

  private def addExcluded(): Unit = {
    excluded.add("com.intel.analytics.bigdl.nn.CellUnit")
    excluded.add("com.intel.analytics.bigdl.nn.tf.ControlDependency")
    excluded.add("com.intel.analytics.bigdl.utils.tf.AdapterForTest")
    excluded.add("com.intel.analytics.bigdl.utils.serializer.TestModule")
    excluded.add("com.intel.analytics.bigdl.utils.ExceptionTest")
  }

  override protected def beforeAll() = {
    addExcluded
    val reflections = new Reflections(new ConfigurationBuilder()
      .filterInputsBy(new FilterBuilder().
        excludePackage("com.intel.analytics.bigdl.utils.tf.loaders"))
      .setUrls(ClasspathHelper.forPackage(pkg))
      .setScanners(new SubTypesScanner()))

    val subTypes = reflections.getSubTypesOf(classOf[AbstractModule[_, _, _]])
      .asScala.filter(sub => !Modifier.isAbstract(sub.getModifiers)).
      filter(sub => !excluded.contains(sub.getName))
    subTypes.foreach(sub => expected.add(sub.getName))
  }

  private def runSerializationTest(module : AbstractModule[_, _, Float],
                                  input : Activity, cls: Class[_] = null) : Unit = {
    val name = module.getName
    val serFile = File.createTempFile(name, postFix)
    val originForward = module.evaluate().forward(input)

    ModulePersister.saveToFile[Float](serFile.getAbsolutePath, null, module.evaluate(), true)
    RNG.setSeed(1000)
    val loadedModule = ModuleLoader.loadFromFile[Float](serFile.getAbsolutePath)

    val afterLoadForward = loadedModule.forward(input)

    if (serFile.exists) {
      serFile.delete
    }

    afterLoadForward should be (originForward)
    if (cls != null) {
      tested.add(cls.getName)
    } else {
      tested.add(module.getClass.getName)
    }
  }

  "Abs serializer" should "work properly" in {
    val abs = Abs[Float, Float]().setName("abs")
    val input = Tensor[Float](5, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(abs, input)
  }

  "ActivityRegularization serializer" should "work properly" in {
    val activityRegularization = ActivityRegularization[Float](l1 = 0.01, l2 = 0.01).
      setName("activityRegularization")
    val input = Tensor[Float](5, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(activityRegularization, input)
  }

  "UpSampling1D serializer" should "work properly" in {
    val upsampling = UpSampling1D[Float](2).setName("upsampling")
    val input = Tensor[Float](2, 5, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(upsampling, input)
  }

  "UpSampling2D serializer" should "work properly" in {
    val upsampling = UpSampling2D[Float](Array(2, 3)).setName("upsampling")
    val input = Tensor[Float](2, 3, 5, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(upsampling, input)
  }

  "Add serializer" should "work properly" in {
    val add = Add[Float](5).setName("add")
    val input = Tensor[Float](5).apply1(_ => Random.nextFloat())
    runSerializationTest(add, input)
  }

  "AddConst serializer" should "work properly" in {
    val addconst = AddConstant[Float](5).setName("addconst")
    val input = Tensor[Float](5).apply1(_ => Random.nextFloat())
    runSerializationTest(addconst, input)
  }

  "BatchNormalization serializer" should "work properly" in {
    val batchNorm = BatchNormalization[Float](5).setName("batchNorm")
    val input = Tensor[Float](2, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(batchNorm, input)
  }

  "BifurcateSplitTable serializer" should "work properly" in {
    val batchNorm = BifurcateSplitTable[Float](1).setName("batchNorm")
    val input = Tensor[Float](2, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(batchNorm, input)
  }

  "BiLinear serializer" should "work properly" in {
    val input1 = Tensor[Float](5, 5).apply1(e => Random.nextFloat())
    val input2 = Tensor[Float](5, 3).apply1(e => Random.nextFloat())
    var input = new Table()
    input(1.toFloat) = input1
    input(2.toFloat) = input2
    val biLinear = Bilinear[Float](5, 3, 2)
    runSerializationTest(biLinear, input)
  }

  "BinaryThreshold serializer" should "work properly" in {
    val binaryThreshold = BinaryThreshold[Float]().setName("binaryThreshold")
    val input = Tensor[Float](2, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(binaryThreshold, input)
  }

  "SpatialDropout1D serializer" should "work properly" in {
    val spatialDropout1D = SpatialDropout1D[Float]()
    val input = Tensor[Float](2, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(spatialDropout1D, input)
  }

  "SpatialDropout2D serializer" should "work properly" in {
    val spatialDropout2D = SpatialDropout2D[Float]()
    val input = Tensor[Float](2, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(spatialDropout2D, input)
  }

  "SpatialDropout3D serializer" should "work properly" in {
    val spatialDropout3D = SpatialDropout3D[Float]()
    val input = Tensor[Float](2, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(spatialDropout3D, input)
  }

  "LocallyConnected1D serializer" should "work properly" in {
    val localyConnected1d =
      LocallyConnected1D[Float](6, 2, outputFrameSize = 2, kernelW = 3, strideW = 1)
    val input = Tensor[Float](6, 2).randn()
    runSerializationTest(localyConnected1d, input)
  }

  "BinaryTreeLSTM serializer" should "work properly" in {

    RNG.setSeed(1000)
    val binaryTreeLSTM = BinaryTreeLSTM[Float](2, 2).setName("binaryTreeLSTM")

    val inputs =
      Tensor[Float](
        T(T(T(1f, 2f),
          T(2f, 3f),
          T(4f, 5f))))

    val tree =
      Tensor[Float](
        T(T(T(2f, 5f, -1f),
          T(0f, 0f, 1f),
          T(0f, 0f, 2f),
          T(0f, 0f, 3f),
          T(3f, 4f, 0f))))

    val input = T(inputs, tree)

    runSerializationTest(binaryTreeLSTM, input)
  }

  "BiRecurrent serializer" should "work properly" in {
    val input = Tensor[Float](1, 5, 6).apply1(e => Random.nextFloat()).transpose(1, 2)
    RNG.setSeed(100)
    val biRecurrent = BiRecurrent[Float]().add(RnnCell[Float](6, 4,
      Sigmoid[Float]())).setName("biRecurrent")
    runSerializationTest(biRecurrent, input)
  }

  "BiRecurrent serializer with BatchNormParams" should "work properly" in {
    val input = Tensor[Float](1, 5, 6).apply1(e => Random.nextFloat()).transpose(1, 2)
    RNG.setSeed(100)
    val biRecurrent = BiRecurrent[Float](batchNormParams =
      BatchNormParams()).add(RnnCell[Float](6, 4, Sigmoid[Float]())).setName("biRecurrentWithNorm")
    runSerializationTest(biRecurrent, input)
  }


  "BiRecurrent serializer" should "work properly with isSplitInput" in {
    val input = Tensor[Float](1, 5, 6).apply1(e => Random.nextFloat()).transpose(1, 2)
    val biRecurrent = BiRecurrent[Float](isSplitInput = false)
      .add(RnnCell[Float](6, 4, Sigmoid[Float]())).setName("biRecurrentWithSplit")
    runSerializationTest(biRecurrent, input)
  }

  "Bottle serializer" should "work properly" in {
    val input = Tensor[Float](10).apply1(e => Random.nextFloat())

    val bottle = new Bottle[Float](Linear[Float](10, 2).
      asInstanceOf[Module[Float]], 2, 2).setName("bottle")
    runSerializationTest(bottle, input)
  }

  "Caddserializer" should "work properly" in {
    val input = Tensor[Float](5, 1).apply1(e => Random.nextFloat())
    val cadd = CAdd[Float](Array(5, 1)).setName("cadd")
    runSerializationTest(cadd, input)
  }

  "CaddTable serializer" should "work properly" in {
    val input1 = Tensor[Float](5, 5).apply1(e => Random.nextFloat())
    val input2 = Tensor[Float](5, 5).apply1(e => Random.nextFloat())
    var input = new Table()
    input(1.toFloat) = input1
    input(2.toFloat) = input2
    val caddTable = CAddTable[Float](false).setName("caddTable")
    runSerializationTest(caddTable, input)
  }

  "CAveTable serializer" should "work properly" in {
    val input1 = Tensor[Float](5, 5).apply1(e => Random.nextFloat())
    val input2 = Tensor[Float](5, 5).apply1(e => Random.nextFloat())
    var input = new Table()
    input(1.toFloat) = input1
    input(2.toFloat) = input2
    val caveTable = CAveTable[Float](false).setName("caveTable")
    runSerializationTest(caveTable, input)
  }

  "CDivTable serializer" should "work properly" in {
    val cdivTable = new CDivTable[Float]().setName("cdivTable")
    val input1 = Tensor[Float](10).apply1(e => Random.nextFloat())
    val input2 = Tensor[Float](10).apply1(e => Random.nextFloat())
    var input = new Table()
    input(1.toFloat) = input1
    input(2.toFloat) = input2
    runSerializationTest(cdivTable, input)
  }

  "Clamp serializer" should "work properly" in {

    val input = Tensor[Float](10).apply1(e => Random.nextFloat())
    val clamp = Clamp[Float, Float](1, 10).setName("clamp")
    runSerializationTest(clamp, input)
  }

  "CMaxTable serializer" should "work properly" in {
    val cmaxTable = new CMaxTable[Float]().setName("cmaxTable")
    val input1 = Tensor[Float](10).apply1(e => Random.nextFloat())
    val input2 = Tensor[Float](10).apply1(e => Random.nextFloat())
    var input = new Table()
    input(1.toFloat) = input1
    input(2.toFloat) = input2
    runSerializationTest(cmaxTable, input)
  }

  "CMinTable serializer" should "work properly" in {
    val cminTable = new CMinTable[Float]().setName("cminTable")
    val input1 = Tensor[Float](10).apply1(e => Random.nextFloat())
    val input2 = Tensor[Float](10).apply1(e => Random.nextFloat())
    var input = new Table()
    input(1.toFloat) = input1
    input(2.toFloat) = input2
    runSerializationTest(cminTable, input)
  }

  "CMul serializer" should "work properly" in {
    val input = Tensor[Float](5, 1).apply1(e => Random.nextFloat())
    val cmul = CMul[Float](Array(5, 1)).setName("cmul")
    runSerializationTest(cmul, input)
  }

  "CMulTable serializer" should "work properly" in {
    val input1 = Tensor[Float](5, 5).apply1(e => Random.nextFloat())
    val input2 = Tensor[Float](5, 5).apply1(e => Random.nextFloat())
    var input = new Table()
    input(1.toFloat) = input1
    input(2.toFloat) = input2

    val cmulTable = CMulTable[Float]().setName("cmulTable")
    runSerializationTest(cmulTable, input)
  }

  "Concatserializer" should "work properly" in {
    val input = Tensor[Float](2, 2, 2).apply1(e => Random.nextFloat())
    val concat = Concat[Float](2).setName("concat")
    concat.add(Abs[Float, Float]())
    concat.add(Abs[Float, Float]())
    runSerializationTest(concat, input)
  }

  "ConcatTable serializer" should "work properly" in {
    val concatTable = new  ConcatTable[Float]().setName("concatTable")
    concatTable.add(Linear[Float](10, 2))
    concatTable.add(Linear[Float](10, 2))
    val input = Tensor[Float](10).apply1(_ => Random.nextFloat())
    runSerializationTest(concatTable, input)
  }

  "Contiguous serializer" should "work properly" in {
    val contiguous = Contiguous[Float]().setName("contiguous")
    val input = Tensor[Float](10).apply1(_ => Random.nextFloat())
    runSerializationTest(contiguous, input)
  }

  "ConvLSTMPeephole2D serializer" should "work properly" in {
    val hiddenSize = 5
    val inputSize = 3
    val seqLength = 4
    val batchSize = 2
    val kernalW = 3
    val kernalH = 3
    val c2d = ConvLSTMPeephole[Float](
      inputSize,
      hiddenSize,
      kernalW, kernalH,
      1,
      withPeephole = false)
    val convLSTMPeephole2d = Recurrent[Float]().setName("convLSTMPeephole2d")
    val model = Sequential[Float]()
      .add(convLSTMPeephole2d
        .add(c2d))
      .add(View[Float](hiddenSize * kernalH * kernalW))

    val input = Tensor[Float](batchSize, seqLength, inputSize, kernalW, kernalH).rand
    runSerializationTest(convLSTMPeephole2d, input, c2d.getClass)
  }

  "ConvLSTMPeephole3D serializer" should "work properly" in {
    val hiddenSize = 5
    val inputSize = 3
    val seqLength = 4
    val batchSize = 2
    val kernalW = 3
    val kernalH = 3
    val c3d = ConvLSTMPeephole3D[Float](
      inputSize,
      hiddenSize,
      kernalW, kernalH,
      1,
      withPeephole = false)
    val convLSTMPeephole3d = Recurrent[Float]().setName("convLSTMPeephole3d")
    val model = Sequential[Float]()
      .add(convLSTMPeephole3d
        .add(c3d))
      .add(View[Float](hiddenSize * kernalH * kernalW))

    val input = Tensor[Float](batchSize, seqLength, inputSize, kernalW, kernalH, 3).rand
    runSerializationTest(convLSTMPeephole3d, input, c3d.getClass)
  }

  "Cosine serializer" should "work properly" in {
    val cosine = Cosine[Float](5, 5).setName("cosine")
    val input = Tensor[Float](5).apply1(_ => Random.nextFloat())
    runSerializationTest(cosine, input)
  }

  "CosineDistance serializer" should "work properly" in {
    val cosineDistance = CosineDistance[Float]().setName("cosineDistance")
    val input1 = Tensor[Float](5, 5).apply1(e => Random.nextFloat())
    val input2 = Tensor[Float](5, 5).apply1(e => Random.nextFloat())
    var input = new Table()
    input(1.toFloat) = input1
    input(2.toFloat) = input2
    runSerializationTest(cosineDistance, input)
  }

  "Cropping2d serializer" should "work properly" in {
    val cropping2d = Cropping2D[Float](Array(2, 2), Array(2, 2), DataFormat.NCHW)
      .setName("Cropping2D")
    val input = Tensor[Float](1, 9, 9, 9).apply1(_ => Random.nextFloat())
    runSerializationTest(cropping2d, input)
  }

  "Cropping3d serializer" should "work properly" in {
    val cropping3d = Cropping3D[Float](Array(2, 2), Array(2, 2), Array(2, 2)).setName("Cropping3D")
    val input = Tensor[Float](1, 9, 9, 9, 9).apply1(_ => Random.nextFloat())
    runSerializationTest(cropping3d, input)
  }

  "CSubTable serializer" should "work properly" in {
    val csubTable = CSubTable[Float]().setName("csubTable")

    val input1 = Tensor[Float](5, 5).apply1(e => Random.nextFloat())
    val input2 = Tensor[Float](5, 5).apply1(e => Random.nextFloat())
    var input = new Table()
    input(1.toFloat) = input1
    input(2.toFloat) = input2
    runSerializationTest(csubTable, input)
  }

  "DenseToSparse serializer" should "work properly" in {
    val denseToSparse = DenseToSparse[Float]().setName("denseToSparse")
    val input = Tensor.range[Float](1, 12, 1)
    runSerializationTest(denseToSparse, input)
  }

  "Dotproduct serializer" should "work properly" in {
    val dotProduct = DotProduct[Float]().setName("dotProduct")
    val input1 = Tensor[Float](5, 5).apply1(e => Random.nextFloat())
    val input2 = Tensor[Float](5, 5).apply1(e => Random.nextFloat())
    var input = new Table()
    input(1.toFloat) = input1
    input(2.toFloat) = input2
    runSerializationTest(dotProduct, input)
  }

  "Dropout serializer" should "work properly" in {
    RNG.setSeed(100)
    val dropout = Dropout[Float]().setName("dropout")
    val input = Tensor[Float](10).apply1(_ => Random.nextFloat())
    runSerializationTest(dropout, input)
  }

  "Echo serializer" should "work properly" in {
    val echo = Echo[Float]().setName("echo")
    val input = Tensor[Float](10).apply1(_ => Random.nextFloat())
    runSerializationTest(echo, input)
  }

  "ELU serializer" should "work properly" in {
    val elu = ELU[Float, Float]().setName("elu")
    val input = Tensor[Float](10).apply1(_ => Random.nextFloat())
    runSerializationTest(elu, input)
  }

  "Euclidena serializer" should "work properly" in {
    val euclidean = Euclidean[Float](7, 7).setName("euclidean")
    val input = Tensor[Float](8, 7).apply1(_ => Random.nextFloat())
    runSerializationTest(euclidean, input)
  }

  "Exp serializer" should "work properly" in {
    val exp = Exp[Float]().setName("exp")
    val input = Tensor[Float](10).apply1(_ => Random.nextFloat())
    runSerializationTest(exp, input)
  }

  "FlattenTable serializer" should "work properly" in {
    val flattenTable = FlattenTable[Float]().setName("flattenTable")
    val input1 = Tensor[Float](5, 5).apply1(e => Random.nextFloat())
    val input2 = Tensor[Float](5, 5).apply1(e => Random.nextFloat())
    var input = new Table()
    input(1.toFloat) = input1
    input(2.toFloat) = input2
    runSerializationTest(flattenTable, input)
  }

  "GaussianDropout serializer" should "work properly" in {
    RNG.setSeed(1000)
    val gaussianDropout = GaussianDropout[Float](0.5).setName("gaussianDropout")
    val input = Tensor[Float](10).apply1(_ => Random.nextFloat())
    runSerializationTest(gaussianDropout, input)
  }

  "GaussianNoise serializer" should "work properly" in {
    RNG.setSeed(1000)
    val gaussianNoise = GaussianNoise[Float](0.5).setName("gaussianNoise")
    val input = Tensor[Float](10).apply1(_ => Random.nextFloat())
    runSerializationTest(gaussianNoise, input)
  }

  "GaussianSampler serializer" should "work properly" in {
    val input1 = Tensor[Float](2, 3).apply1(x => RNG.uniform(0, 1).toFloat)
    val input2 = Tensor[Float](2, 3).apply1(x => RNG.uniform(0, 1).toFloat)
    val input = T(input1, input2)
    RNG.setSeed(1000)
    val gaussianSampler = GaussianSampler[Float]().setName("gaussianSampler")
    runSerializationTest(gaussianSampler, input)
  }

  "GradientReversal serializer" should "work properly" in {
    val gradientReversal = GradientReversal[Float]().setName("gradientReversal")
    val input = Tensor[Float](10).apply1(_ => Random.nextFloat())
    runSerializationTest(gradientReversal, input)
  }

  "Graph serializer" should "work properly" in {
    val linear = Linear[Float](10, 2).inputs()
    val graph = Graph[Float](linear, linear).setName("graph")
    val input = Tensor[Float](10).apply1(_ => Random.nextFloat())
    runSerializationTest(graph, input)
  }

  "Graph with variables serializer" should "work properly" in {
    val linear = Linear[Float](2, 2)
    val linearNode = linear.inputs()
    val linearWeight = linear.weight
    val linearBias = linear.bias
    val variables = Some(Array(linearWeight), Array(linearBias))
    val graphWithVariable = Graph[Float](Array(linearNode), Array(linearNode),
      variables).setName("graphWithVariable")
    val input = Tensor[Float](2).apply1(_ => Random.nextFloat())
    runSerializationTest(graphWithVariable, input)
  }

  "Dynamic Graph with variables serializer" should "work properly" in {
    val linear = Linear[Float](2, 2)
    val linearNode = linear.inputs()
    val linearWeight = linear.weight
    val linearBias = linear.bias
    val variables = Some(Array(linearWeight), Array(linearBias))
    val graphWithVariable = Graph.dynamic[Float](Array(linearNode), Array(linearNode),
      variables, false).setName("graphWithVariable")
    val input = Tensor[Float](2).apply1(_ => Random.nextFloat())
    runSerializationTest(graphWithVariable, input)
  }

  "Dynamic Graph with control ops serializer" should "work properly" in {
    val data = Input[Float]("data")
    val condition = Input[Float]("condition")
    val swtich = ControlNodes.switch(condition, data)
    val echo1 = Echo[Float]().inputs(swtich.trueEdge())
    val echo2 = Echo[Float]().inputs(swtich.falseEdge())

    val model = Graph.dynamic[Float](Array(data, condition), Array(echo1), None, false)

    val input = T(Tensor[Float](T(1)), Tensor[Boolean](T(true)))

    runSerializationTest(model, input)
  }

  "GRU serializer" should "work properly" in {
    RNG.setSeed(100)
    val gru = GRU[Float](100, 100)
    val gruModel = Recurrent[Float]().add(gru).setName("gru")
    val input = Tensor[Float](2, 20, 100).apply1(e => Random.nextFloat())
    runSerializationTest(gruModel, input, gru.getClass)
  }

  "HardShrink serializer" should "work properly" in {
    val hardShrink = HardShrink[Float]().setName("hardShrink")
    val input = Tensor[Float](10).apply1(_ => Random.nextFloat())
    runSerializationTest(hardShrink, input)
  }

  "HardTanh serializer" should "work properly" in {
    val hardTanh = HardTanh[Float, Float]().setName("hardTanh")
    val input = Tensor[Float](10).apply1(_ => Random.nextFloat())
    runSerializationTest(hardTanh, input)
  }

  "HardSigmoid serialization" should "work properly" in {
    val hardSigmoid = HardSigmoid[Float]().setName("hardSigmoid")
    val input = Tensor[Float](2, 2).rand()
    runSerializationTest(hardSigmoid, input)
  }

  "Identity serializer" should "work properly" in {
    val identity = Identity[Float]().setName("identity")
    val input = Tensor[Float](10).apply1(_ => Random.nextFloat())
    runSerializationTest(identity, input)
  }

  "Index serializer" should "work properly" in {
    val index = Index[Float](1).setName("index")
    val input1 = Tensor[Float](3).apply1(e => Random.nextFloat())
    val input2 = Tensor[Float](4)
    input2(Array(1)) = 1
    input2(Array(2)) = 2
    input2(Array(3)) = 2
    input2(Array(4)) = 3
    val input = new Table()
    input(1.toFloat) = input1
    input(2.toFloat) = input2
    runSerializationTest(index, input)
  }

  "InferReshape serializer" should "work properly" in {
    val inferReshape = InferReshape[Float](Array(-1, 2, 0, 5)).setName("inferReshape")
    val input = Tensor[Float](2, 5, 2, 2).apply1(_ => Random.nextFloat())
    runSerializationTest(inferReshape, input)
  }

  "Input serializer" should "work properly " in {
    val inputl = Input[Float]().element.setName("input")
    val input = Tensor[Float](10).apply1(_ => Random.nextFloat())
    runSerializationTest(inputl, input)
  }

  "JoinTable serializer" should "work  properly" in {
    val joinTable = JoinTable[Float](2, 2).setName("joinTable")
    val input1 = Tensor[Float](2, 2).apply1(_ => Random.nextFloat())
    val input2 = Tensor[Float](2, 2).apply1(_ => Random.nextFloat())
    val input = T()
    input(1.toFloat) = input1
    input(2.toFloat) = input2
    runSerializationTest(joinTable, input)

  }

  "L1Penalty serializer" should "work properly" in {
    val l1Penalty = L1Penalty[Float](1, true, true).setName("l1Penalty")
    val input = Tensor[Float](3, 3).apply1(_ => Random.nextFloat())
    runSerializationTest(l1Penalty, input)
  }

  "LeakReLu serializer" should  "work properly" in {
    val leakyReLU = LeakyReLU[Float](0.01, true).setName("leakyReLU")
    val input = Tensor[Float](3, 3).apply1(_ => Random.nextFloat())
    runSerializationTest(leakyReLU, input)
  }

  "Linear serializer" should "work properly" in {
    val linear = Linear[Float](10, 2).setName("linear")
    val input = Tensor[Float](10).apply1(_ => Random.nextFloat())
    runSerializationTest(linear, input)
  }

  "Log Serializer" should "work properly" in {
    val log = Log[Float, Float]().setName("log")
    val input = Tensor[Float](10).apply1(_ => Random.nextFloat())
    runSerializationTest(log, input)
  }

  "LogSigmoid serializer" should "work properly" in {
    val logSigmoid = LogSigmoid[Float]().setName("logSigmoid")
    val input = Tensor[Float](10).apply1(_ => Random.nextFloat())
    runSerializationTest(logSigmoid, input)
  }

  "LogSogMax serializer" should "work properly" in {
    val logSoftMax = LogSoftMax[Float]().setName("logSoftMax")
    val input = Tensor[Float](10).apply1(_ => Random.nextFloat())
    runSerializationTest(logSoftMax, input)
  }

  "LookupTable serializer" should "work properly" in {
    val lookupTable = LookupTable[Float](9, 4, 2, 0.1, 2.0, true).setName("lookupTable")
    val input = Tensor[Float](5)
    input(Array(1)) = 5
    input(Array(2)) = 2
    input(Array(3)) = 6
    input(Array(4)) = 9
    input(Array(5)) = 4
    runSerializationTest(lookupTable, input)
  }

  "LSTM serializer" should "work properly" in {
    val lstm = LSTM[Float](6, 4)
    val lstmModel = Recurrent[Float]().add(lstm).setName("lstm")
    val input = Tensor[Float](Array(1, 5, 6)).apply1(_ => Random.nextFloat())
    runSerializationTest(lstmModel, input, lstm.getClass)
  }

  "LookupTableSparse serializer" should "work properly" in {
    val lookupTableSparse = LookupTableSparse[Float](20, 10, "sum", 1)
    val indices1 = Array(0, 0, 1, 2)
    val indices2 = Array(0, 1, 0, 3)
    val values = Array(2f, 4, 1, 2)
    val input = Tensor.sparse[Float](Array(indices1, indices2), values, Array(3, 4))
    runSerializationTest(lookupTableSparse, input, lookupTableSparse.getClass)
  }

  "LSTMPeephole serializer" should "work properly" in {
    val lstmPeephole = LSTMPeephole[Float](6, 4)
    val lstmPeepholeModel = Recurrent[Float]().add(lstmPeephole).setName("lstmPeephole")
    val input = Tensor[Float](Array(1, 5, 6)).apply1(_ => Random.nextFloat())
    runSerializationTest(lstmPeepholeModel, input, lstmPeephole.getClass)
  }

  "MapTable serializer" should "work properly" in {
    val linear = Linear[Float](2, 2)
    val mapTable = new MapTable[Float]().setName("mapTable")
    mapTable.add(linear)
    val input1 = Tensor[Float](2).apply1(_ => Random.nextFloat())
    val input2 = Tensor[Float](2).apply1(_ => Random.nextFloat())
    val input = T()
    input(1.0.toFloat) = input1
    input(2.0.toFloat) = input2
    runSerializationTest(mapTable, input)
  }

  "MaskedSelect serializer" should "work properly" in {
    val maskedSelect = MaskedSelect[Float]().setName("maskedSelect")
    val input1 = Tensor[Float](2, 2).apply1(e => Random.nextFloat())
    val input2 = Tensor[Float](2, 2)
    input2(Array(1, 1)) = 1
    input2(Array(1, 2)) = 0
    input2(Array(2, 1)) = 0
    input2(Array(2, 2)) = 1
    val input = new Table()
    input(1.0f) = input1
    input(2.0f) = input2
    runSerializationTest(maskedSelect, input)
  }

  "Masking serializer" should "work properly" in {
    val masking = Masking[Float](0.1).setName("masking")
    val input = Tensor[Float](2, 3, 4).apply1(_ => Random.nextFloat())
    runSerializationTest(masking, input)
  }

  "Max serializer" should "work properly" in {
    val max = new Max[Float](2).setName("max")
    val input = Tensor[Float](2, 3, 4).apply1(_ => Random.nextFloat())
    runSerializationTest(max, input)
  }

  "Maxout serializer" should "work properly" in {
    val maxout = Maxout[Float](2, 4, 5).setName("maxout")
    val input = Tensor[Float](2).apply1(_ => Random.nextFloat())
    runSerializationTest(maxout, input)
  }

  "Mean serializer" should "work properly " in {
    val mean = Mean[Float](2).setName("mean")
    val input = Tensor[Float](5, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(mean, input)
  }

  "Min serializer" should "work properly " in {
    val min = Min[Float](2).setName("min")
    val input = Tensor[Float](5, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(min, input)
  }

  "MixtureTable Serializer" should "work properly " in {
    val mixTureTable = MixtureTable[Float]().setName("mixTureTable")
    val input1 = Tensor[Float](2, 2).apply1(e => Random.nextFloat())
    val input2 = Tensor[Float](2, 2).apply1(e => Random.nextFloat())
    val input = new Table()
    input(1.0f) = input1
    input(2.0f) = input2
    runSerializationTest(mixTureTable, input)
  }

  "MM Serializer" should "work properly" in {
    val mm = MM[Float]().setName("mm_layer")
    val input1 = Tensor[Float](2, 3).apply1(e => Random.nextFloat())
    val input2 = Tensor[Float](3, 4).apply1(e => Random.nextFloat())
    val input = new Table()
    input(1.0f) = input1
    input(2.0f) = input2
    runSerializationTest(mm, input)
  }

  "Mul Serializer" should "work properly" in {
    val mul = Mul[Float]().setName("mul")
    val input = Tensor[Float](10, 10).apply1(_ => Random.nextFloat())
    runSerializationTest(mul, input)
  }

  "MulConst Serializer" should "work properly" in {
    val mulConst = MulConstant[Float](1.0).setName("mulConst")
    val input = Tensor[Float](10, 10).apply1(_ => Random.nextFloat())
    runSerializationTest(mulConst, input)
  }

  "MultiRNNCell serializer" should "work properly" in {
    val hiddenSize = 5
    val inputSize = 5
    val seqLength = 4
    val batchSize = 2
    val kernalW = 3
    val kernalH = 3
    val rec = RecurrentDecoder[Float](seqLength)
    val cells = Array(ConvLSTMPeephole[Float](
      inputSize,
      hiddenSize,
      kernalW, kernalH,
      1), ConvLSTMPeephole[Float](
      inputSize,
      hiddenSize,
      kernalW, kernalH,
      1), ConvLSTMPeephole[Float](
      inputSize,
      hiddenSize,
      kernalW, kernalH,
      1)).asInstanceOf[Array[Cell[Float]]]

    val multiRNNCell = MultiRNNCell[Float](cells)

    val model = Sequential[Float]()
      .add(rec
        .add(multiRNNCell)).setName("multiRNNCell")

    val input = Tensor[Float](batchSize, inputSize, 10, 10).apply1(_ => Random.nextFloat())
    runSerializationTest(model, input, multiRNNCell.getClass)
  }

  "MV Serializer" should "work properly" in {
    val mv = MV[Float]().setName("mv_layer")
    val input1 = Tensor[Float](2, 3).apply1(e => Random.nextFloat())
    val input2 = Tensor[Float](3).apply1(e => Random.nextFloat())
    val input = new Table()
    input(1.0f) = input1
    input(2.0f) = input2
    runSerializationTest(mv, input)
  }

  "Narrow serializer" should "work properly" in {
    val narrow = Narrow[Float](1, 3, -3).setName("narrow")
    val input = Tensor[Float](9, 4, 14).apply1(e => Random.nextFloat())
    runSerializationTest(narrow, input)
  }

  "NarrowTable serializer" should "work properly" in {
    val narrowTable = NarrowTable[Float](1, 1)
    val input = T()
    input(1.0) = Tensor[Float](2, 2).apply1(e => Random.nextFloat())
    input(2.0) = Tensor[Float](2, 2).apply1(e => Random.nextFloat())
    input(3.0) = Tensor[Float](2, 2).apply1(e => Random.nextFloat())
    runSerializationTest(narrowTable, input)
  }

  "Negative serializer" should "work properly" in {
    val negative = Negative[Float]().setName("negative")
    val input = Tensor[Float](10).apply1(e => Random.nextFloat())
    runSerializationTest(negative, input)
  }

  "Normlize serializer" should "work properly" in {
    val normalizer = Normalize[Float](2).setName("normalizer")
    val input = Tensor[Float](2, 3, 4, 4).apply1(e => Random.nextFloat())
    runSerializationTest(normalizer, input)
  }

  "Pack serializer" should "work properly" in {
    val pack = new Pack[Float](1).setName("pack")
    val input1 = Tensor[Float](2, 2).apply1(e => Random.nextFloat())
    val input2 = Tensor[Float](2, 2).apply1(e => Random.nextFloat())
    val input = T()
    input(1.0f) = input1
    input(2.0f) = input2
    runSerializationTest(pack, input)
  }

  "Padding serializer" should "work properly" in {
    val padding = Padding[Float](1, -1, 4, -0.8999761, 14).setName("padding")
    val input = Tensor[Float](3, 13, 11).apply1(e => Random.nextFloat())
    runSerializationTest(padding, input)
  }

  "PairwiseDistance serializer" should "work properly" in {
    val pairwiseDistance = new PairwiseDistance[Float](3).setName("pairwiseDistance")
    val input1 = Tensor[Float](3, 3).apply1(e => Random.nextFloat())
    val input2 = Tensor[Float](3, 3).apply1(e => Random.nextFloat())
    val input = T(1.0f -> input1, 2.0f -> input2)
    runSerializationTest(pairwiseDistance, input)
  }

  "ParallelTable serializer" should "work properly" in {
    val parallelTable = ParallelTable[Float]().setName("parallelTable")
    parallelTable.add(Linear[Float](2, 2))
    parallelTable.add(Linear[Float](2, 2))
    val input1 = Tensor[Float](2, 2).apply1(e => Random.nextFloat())
    val input2 = Tensor[Float](2, 2).apply1(e => Random.nextFloat())
    val input = T(1.0f -> input1, 2.0f -> input2)
    runSerializationTest(parallelTable, input)
  }

  "Power serializer" should "work properly" in {
    val power = Power[Float, Float](2.0).setName("power")
    val input = Tensor[Float](2, 2).apply1(e => Random.nextFloat())
    runSerializationTest(power, input)
  }

  "PReLU serializer" should "work properly" in {
    val preLu = PReLU[Float](2).setName("preLu")
    val input = Tensor[Float](2, 3, 4).apply1(_ => Random.nextFloat())
    runSerializationTest(preLu, input)
  }

  "Recurrent serializer" should "work properly" in {
    val recurrent = Recurrent[Float]().setName("recurrent")
      .add(RnnCell[Float](5, 4, Tanh[Float]()))
    val input = Tensor[Float](Array(10, 5, 5)).apply1(_ => Random.nextFloat())
    runSerializationTest(recurrent, input)
  }

  "Recurrent serializer" should "work properly with BatchNormParams" in {
    val recurrent = Recurrent[Float](BatchNormParams()).setName("recurrentWithNorm")
      .add(RnnCell[Float](5, 4, Tanh[Float]()))
    val input = Tensor[Float](Array(10, 5, 5)).apply1(_ => Random.nextFloat())
    runSerializationTest(recurrent, input)
  }

  "RecurrentDecoder serializer" should "work properly" in {
    val recDecoder = RecurrentDecoder[Float](5).
      add(ConvLSTMPeephole[Float](7, 7, 3, 3, 1))
    val input = Tensor[Float](4, 7, 5, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(recDecoder, input)
  }

  "ReLU serializer" should "work properly" in {
    val relu = ReLU[Float]().setName("relu")
    val input = Tensor[Float](5, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(relu, input)
  }

  "ReLU6 serializer" should "work properly" in {
    val relu6 = ReLU6[Float, Float](false).setName("relu6")
    val input = Tensor[Float](10).apply1(_ => Random.nextFloat())
    runSerializationTest(relu6, input)
  }

  "Replicate serializer" should "work properly" in {
    val replicate = new Replicate[Float](3).setName("replicate")
    val input = Tensor[Float](10).apply1(_ => Random.nextFloat())
    runSerializationTest(replicate, input)
  }

  "Reshape serializer" should "work properly" in {
    val reshape = Reshape[Float](Array(1, 4, 5)).setName("reshape")
    val input = Tensor[Float](2, 2, 5).apply1( _ => Random.nextFloat())
    runSerializationTest(reshape, input)
  }

  "ResizeBilinear serializer" should "work properly" in {
    val input = Tensor[Float](1, 3, 2, 3).apply1( _ => Random.nextFloat())
    val resizeBilinear = ResizeBilinear[Float](3, 2).setName("resizeBilinear")
    runSerializationTest(resizeBilinear, input)
  }

  "Reverse serializer" should "work properly" in {
    val reverse = Reverse[Float]().setName("reverse")
    val input = Tensor[Float](10).apply1(_ => Random.nextFloat())
    runSerializationTest(reverse, input)
  }

  "RnnCell serializer" should "work properly" in {
    val rnnCell = RnnCell[Float](6, 4, Sigmoid[Float]()).setName("rnnCell")
    val input1 = Tensor[Float](Array(1, 4)).apply1(_ => Random.nextFloat())
    val input2 = Tensor[Float](Array(1, 4)).apply1(_ => Random.nextFloat())
    val input = T()
    input(1.0f) = input1
    input(2.0f) = input2
    runSerializationTest(rnnCell, input)
  }

  "RoiPooling serializer" should " work properly" in {
    val input = T()
    val input1 = Tensor[Float](1, 1, 2, 2).apply1(_ => Random.nextFloat())
    val input2 = Tensor[Float](1, 5).apply1(_ => Random.nextFloat())
    input(1.0f) = input1
    input(2.0f) = input2
    val roiPooling = new RoiPooling[Float](pooledW = 3,
      pooledH = 2, 1.0f).setName("roiPooling")
    runSerializationTest(roiPooling, input)
  }

  "RReLU serializer" should "work properly" in {
    val rrelu = new RReLU[Float](inplace = false).setName("rrelu")
    val input = Tensor[Float](2, 2, 2).apply1(_ => Random.nextFloat())
    runSerializationTest(rrelu, input)
  }

  "Scale serializer" should "work properly" in {
    val scale = Scale[Float](Array(1, 4, 1, 1)).setName("scale")
    val input = Tensor[Float](1, 4, 5, 6).apply1(_ => Random.nextFloat())
    runSerializationTest(scale, input)
  }

  "Select serializer" should "work properly" in {
    val select = Select[Float](2, 2).setName("select")
    val input = Tensor[Float](5, 5, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(select, input)
  }

  "SelectTable serializer" should "work properly" in {
    val selectTable = SelectTable[Float](2).setName("selectTable")
    val input1 = Tensor[Float](10).apply1(_ => Random.nextFloat())
    val input2 = Tensor[Float](10).apply1(_ => Random.nextFloat())
    val input3 = Tensor[Float](10).apply1(_ => Random.nextFloat())
    val input = T(1.0 -> input1, 2.0 -> input2, 3.0 -> input3)
    runSerializationTest(selectTable, input)
  }

  "Sequential Container" should "work properly" in {
    val sequential = Sequential[Float]().setName("sequential")
    val linear = Linear[Float](10, 2)
    sequential.add(linear)
    val input = Tensor[Float](10).apply1(_ => Random.nextFloat())
    runSerializationTest(sequential, input)
  }

  "Sigmoid serializer" should "work properly" in {
    val sigmoid = Sigmoid[Float]().setName("sigmoid")
    val input = Tensor[Float](10).apply1(_ => Random.nextFloat())
    runSerializationTest(sigmoid, input)
  }

  "SoftMax serializer" should  "work properly" in {
    val softMax = SoftMax[Float]().setName("softMax")
    val input = Tensor[Float](10).apply1(_ => Random.nextFloat())
    runSerializationTest(softMax, input)
  }

  "SoftMin serializer" should "work properly" in {
    val softMin = SoftMin[Float]().setName("softMin")
    val input = Tensor[Float](10).apply1(_ => Random.nextFloat())
    runSerializationTest(softMin, input)
  }

  "SoftPlus serializer" should "work properly" in {
    val softPlus = SoftPlus[Float, Float]().setName("softPlus")
    val input = Tensor[Float](10).apply1(_ => Random.nextFloat())
    runSerializationTest(softPlus, input)
  }

  "SoftShrink serializer" should "work properly" in {
    val softShrink = SoftShrink[Float]().setName("softShrink")
    val input = Tensor[Float](10, 10).apply1(_ => Random.nextFloat())
    runSerializationTest(softShrink, input)
  }

  "SoftSign serializer" should "work properly" in {
    val softSign = SoftSign[Float, Float]().setName("softSign")
    val input = Tensor[Float](10, 10).apply1(_ => Random.nextFloat())
    runSerializationTest(softSign, input)
  }

  "SparseJoinTable serializer" should "work properly" in {
    val sparseJoinTable = SparseJoinTable[Float](2).setName("sparseJoinTable")
    val sparseModel = Sequential[Float]().
      add(ParallelTable[Float]().add(Identity[Float]()).add(Identity[Float]()))
      .add(sparseJoinTable)
    val input1 = Tensor[Float](4, 3).apply1(_ => Random.nextInt(2) * Random.nextFloat())
    val input2 = Tensor[Float](4, 2).apply1(_ => Random.nextInt(2) * Random.nextFloat())
    val sparseInput = T(Tensor.sparse(input1), Tensor.sparse(input2))
    runSerializationTest(sparseJoinTable, sparseInput)
  }

  "SparseLinear serializer" should "work properly" in {
    val sparseLinear = SparseLinear[Float](4, 2).setName("sparseLinear")
    val input = Tensor[Float](2, 4).apply1(_ => Random.nextFloat())
    val sparseInput = Tensor.sparse(input)
    runSerializationTest(sparseLinear, sparseInput)
  }

  "SpatialAveragePooling serializer" should "work properly" in {
    val spatialAveragePooling = new SpatialAveragePooling[Float](3, 2, 2, 1).
      setName("spatialAveragePooling")
    val input = Tensor[Float](1, 4, 3).apply1(_ => Random.nextFloat())
    runSerializationTest(spatialAveragePooling, input)
  }

  "SpatialBatchNormalization serializer" should "work properly" in {
    val spatialBatchNorm = SpatialBatchNormalization[Float](5).
      setName("spatialBatchNorm")
    val input = Tensor[Float](2, 5, 4, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(spatialBatchNorm, input)
  }

  "SpatialContrastiveNormalization serializer" should "work properly" in {
    RNG.setSeed(100)
    val spatialContrastiveNorm = new SpatialContrastiveNormalization[Float]().
      setName("spatialContrastiveNorm")
    val input = Tensor[Float](1, 5, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(spatialContrastiveNorm, input)
  }

  "SpatialConvolution serializer" should "work properly" in {
    val spatialConvolution = SpatialConvolution[Float](3, 4, 2, 2).
      setName("spatialConvolution")
    val input = Tensor[Float](1, 3, 5, 5).apply1( e => Random.nextFloat())
    runSerializationTest(spatialConvolution, input)
  }

  "LocallyConnected2D serializer" should "work properly" in {
    val locallyConnected2D = LocallyConnected2D[Float](3, 5, 5, 4, 2, 2).
      setName("locallyConnected2D")
    val input = Tensor[Float](1, 3, 5, 5).apply1( e => Random.nextFloat())
    runSerializationTest(locallyConnected2D, input)
  }

  "SpatialConvolutionMap serializer" should "work properly" in {
    val spatialConvolutionMap = SpatialConvolutionMap[Float](
      SpatialConvolutionMap.random(1, 1, 1), 2, 2).setName("spatialConvolutionMap")
    val input = Tensor[Float](1, 3, 3).apply1( e => Random.nextFloat())
    runSerializationTest(spatialConvolutionMap, input)
  }

  "SpatialCrossMapLRN serializer" should "work properly" in {
    val spatialCrossMapLRN = SpatialCrossMapLRN[Float](5, 0.01, 0.75, 1.0).
      setName("spatialCrossMapLRN")
    val input = Tensor[Float](2, 2, 2, 2).apply1( e => Random.nextFloat())
    runSerializationTest(spatialCrossMapLRN, input)
  }

  "SpatialDilatedConvolution serializer" should "work properly" in {

    val spatialDilatedConvolution = SpatialDilatedConvolution[Float](1, 1,
      2, 2, 1, 1, 0, 0).setName("spatialDilatedConvolution")
    val input = Tensor[Float](1, 3, 3).apply1( e => Random.nextFloat())
    runSerializationTest(spatialDilatedConvolution, input)
  }

  "SpatialDivisiveNormalization serializer" should "work properly" in {
    val spatialDivisiveNormalization = SpatialDivisiveNormalization[Float]().
      setName("spatialDivisiveNormalization")
    val input = Tensor[Float](1, 5, 5).apply1(e => Random.nextFloat())
    runSerializationTest(spatialDivisiveNormalization, input)
  }

  "SpatialFullConvolution serializer" should "work properly" in {

    val spatialFullConvolution = SpatialFullConvolution[Float](1, 1,
      2, 2, 1, 1, 0, 0).setName("spatialFullConvolution")
    val input = Tensor[Float](1, 3, 3).apply1(e => Random.nextFloat())
    runSerializationTest(spatialFullConvolution, input)
  }

  "SpatialMaxPooling serializer" should "work properly" in {
    val spatialMaxPooling = SpatialMaxPooling[Float](2, 2, 2, 2).
      setName("spatialMaxPooling")
    val input = Tensor[Float](1, 3, 3).apply1( e => Random.nextFloat())
    runSerializationTest(spatialMaxPooling, input)
  }

  "SpatialShareConvolution serializer" should "work properly" in {
    val spatialShareConvolution = SpatialShareConvolution[Float](1, 1, 2, 2, 1, 1).
      setName("spatialShareConvolution")
    val input = Tensor[Float](3, 1, 3, 4).apply1( e => Random.nextFloat())
    runSerializationTest(spatialShareConvolution, input)
  }

  "SpatialSubtractiveNormalization serializer" should "work properly" in {
    val kernel = Tensor[Float](3, 3).apply1( e => Random.nextFloat())
    val spatialSubtractiveNormalization = SpatialSubtractiveNormalization[Float](1, kernel).
      setName("spatialSubtractiveNormalization")
    val input = Tensor[Float](1, 1, 1, 5).apply1( e => Random.nextFloat())
    runSerializationTest(spatialSubtractiveNormalization, input)
  }

  "SpatialWithinChannelLRN serializer" should "work properly" in {
    val spatialWithinChannelLRN = new SpatialWithinChannelLRN[Float](5, 5e-4, 0.75).
      setName("spatialWithinChannelLRN")
    val input = Tensor[Float](1, 4, 7, 6).apply1( e => Random.nextFloat())
    runSerializationTest(spatialWithinChannelLRN, input)
  }

  "SpatialZeroPadding serializer" should "work properly" in {
    val spatialZeroPadding = SpatialZeroPadding[Float](1, 0, -1, 0).
      setName("spatialZeroPadding")
    val input = Tensor[Float](3, 3, 3).apply1(_ => Random.nextFloat())
    runSerializationTest(spatialZeroPadding, input)
  }

  "SpatialSeperableConvolution serializer" should "work properly" in {
    val seprableConv = SpatialSeperableConvolution[Float](2, 2, 1, 2, 2,
      dataFormat = DataFormat.NHWC).setName("seprableConv")
    val input = Tensor[Float](1, 5, 5, 2).apply1( e => Random.nextFloat())
    runSerializationTest(seprableConv, input)
  }

  "SplitTable serializer" should "work properly" in {
    val splitTable = SplitTable[Float](2).setName("splitTable")
    val input = Tensor[Float](2, 10).apply1( e => Random.nextFloat())
    runSerializationTest(splitTable, input)
  }

  "Sqrt serializer" should "work properly" in {
    val sqrt = Sqrt[Float, Float]().setName("sqrt")
    val input = Tensor[Float](10).apply1( e => Random.nextFloat())
    runSerializationTest(sqrt, input)
  }

  "Square serializer" should "work properly" in {
    val square = Square[Float, Float]().setName("square")
    val input = Tensor[Float](10).apply1( e => Random.nextFloat())
    runSerializationTest(square, input)
  }

  "Squeeze serializer" should "work properly" in {
    val squeeze = Squeeze[Float](2).setName("squeeze")
    val input = Tensor[Float](2, 1, 2).apply1( e => Random.nextFloat())
    runSerializationTest(squeeze, input)
  }

  "SReLU serilalizer" should "work properly" in {
    val srelu = SReLU[Float]().setName("srelu")
    val input = Tensor[Float](3, 4).apply1( e => Random.nextFloat())
    runSerializationTest(srelu, input)
  }

  "Sum serializer" should "work properly" in {
    val sum = Sum[Float, Float](2).setName("sum")
    val input = Tensor[Float](5, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(sum, input)
  }

  "Tanh serializer" should "work properly" in {
    val tanh = Tanh[Float]().setName("tanh")
    val input = Tensor[Float](5, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(tanh, input)
  }

  "TanhShrink serializer" should "work properly" in {
    val tanhShrink = TanhShrink[Float]().setName("tanhShrink")
    val input = Tensor[Float](5, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(tanhShrink, input)
  }

  "TemporalConvolution serializer" should "work properly" in {
    val temporalConvolution = TemporalConvolution[Float](10, 8, 5, 2).
      setName("temporalConvolution")
    val input = Tensor[Float](100, 10).apply1(e => Random.nextFloat())
    runSerializationTest(temporalConvolution, input)
  }

  "TemporalMaxPooling serializer" should "work properly" in {
    val temporalMaxPooling = new TemporalMaxPooling[Float](4).setName("temporalMaxPooling")
    val input = Tensor[Float](5, 4, 5).apply1(e => Random.nextFloat())
    runSerializationTest(temporalMaxPooling, input)
  }

  "Threshold serializer" should "work properly" in {
    val threshold = Threshold[Float](0.5).setName("threshold")
    val input = Tensor[Float](5, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(threshold, input)
  }

  "Tile serializer" should "work properly" in {
    val tile = Tile[Float](1).setName("tile")
    val input = Tensor[Float](5, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(tile, input)
  }

  "TimeDistributed serializer" should "work properly" in {
    val timeDistributed = TimeDistributed[Float](Linear[Float](5, 5)).
      setName("timeDistributed")
    val input = Tensor[Float](2, 5, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(timeDistributed, input)
  }

  "Transpose serializer" should "work properly" in {
    val transpose = Transpose[Float](Array((1, 2))).setName("transpose")
    val input = Tensor[Float]().resize(Array(2, 3)).apply1(_ => Random.nextFloat())
    runSerializationTest(transpose, input)
  }

  "Unsqueeze serializer" should "work properly" in {
    val unsqueeze = Unsqueeze[Float](2).setName("unsqueeze")
    val input = Tensor[Float](2, 2, 2).apply1(_ => Random.nextFloat())
    runSerializationTest(unsqueeze, input)
  }

  "UpSampling3D serializer" should "work properly" in {
    val upSampling3D = UpSampling3D[Float](Array(2, 2, 2)).setName("upSampling3D")
    val input = Tensor[Float](1, 2, 2, 2, 2).apply1(_ => Random.nextFloat())
    runSerializationTest(upSampling3D, input)
  }

  "View serializer" should "work properly" in {
    val view = View[Float](Array(2, 5)).setName("view")
    val input = Tensor[Float](1, 10).apply1(_ => Random.nextFloat())
    runSerializationTest(view, input)
  }

  "VolumetricAveragePooling serializer" should "work properly" in {
    val volumetricAveragePooling = VolumetricAveragePooling[Float](2, 2, 2, 1, 1, 1, 0, 0, 0).
        setName("volumetricAveragePooling")
    val input = Tensor[Float](1, 2, 3, 3).apply1(_ => Random.nextFloat())
    runSerializationTest(volumetricAveragePooling, input)
  }

  "VolumetricConvolution serializer" should "work properly" in {
    val volumetricConvolution = VolumetricConvolution[Float](2, 3, 2, 2, 2, dT = 1, dW = 1, dH = 1,
      padT = 0, padW = 0, padH = 0, withBias = true).setName("volumetricConvolution")
    val input = Tensor[Float](2, 2, 2, 2).apply1(_ => Random.nextFloat())
    runSerializationTest(volumetricConvolution, input)
  }

  "VolumetricFullConvolution serializer" should "work properly" in {

    val volumetricFullConvolution = new VolumetricFullConvolution[Float](3, 6,
      4, 3, 3, 2, 1, 1, 2, 2, 2).setName("volumetricFullConvolution")
    val input = Tensor[Float](3, 3, 3, 6, 6).apply1(e => Random.nextFloat())
    runSerializationTest(volumetricFullConvolution, input)
  }

  "VolumetricMaxPooling serializer" should "work properly" in {
    val volumetricMaxPooling = VolumetricMaxPooling[Float](2, 2, 2, 1, 1, 1, 0, 0, 0).
      setName("volumetricMaxPooling")
    val input = Tensor[Float](1, 2, 3, 3).apply1(_ => Random.nextFloat())
    runSerializationTest(volumetricMaxPooling, input)
  }

  "bigquant.SpatialConvolution serializer" should "work properly" in {
    val nInputPlane = 1
    val nOutputPlane = 1
    val kW = 2
    val kH = 2
    val dW = 1
    val dH = 1
    val padW = 0
    val padH = 0

    val kernelData = Array(
      2.0f, 3f,
      4f, 5f
    )

    val biasData = Array(0.0f)

    val input = Tensor[Float](1, 1, 3, 3).apply1(_ => Random.nextFloat())
    val weight = Tensor[Float](Storage(kernelData), 1, Array(nOutputPlane, nInputPlane, kH, kW))
    val bias = Tensor[Float](Storage(biasData), 1, Array(nOutputPlane))
    val conv = quantized.SpatialConvolution[Float](nInputPlane, nOutputPlane,
      kW, kH, dW, dH, padW, padH, initWeight = weight, initBias = bias).setName("quantConv")

    runSerializationTest(conv, input)
  }

  "bigquant.SpatialDilatedConvolution serializer" should "work properly" in {
    val nInputPlane = 1
    val nOutputPlane = 1
    val kW = 2
    val kH = 2
    val dW = 1
    val dH = 1
    val padW = 0
    val padH = 0

    val kernelData = Array(
      2.0f, 3f,
      4f, 5f
    )

    val biasData = Array(0.0f)

    val input = Tensor[Float](1, 1, 3, 3).apply1(_ => Random.nextFloat())
    val weight = Tensor[Float](Storage(kernelData), 1, Array(nOutputPlane, nInputPlane, kH, kW))
    val bias = Tensor[Float](Storage(biasData), 1, Array(nOutputPlane))
    val conv = quantized.SpatialDilatedConvolution[Float](nInputPlane, nOutputPlane,
      kW, kH, dW, dH, padW, padH, initWeight = weight, initBias = bias)
      .setName("quantDilatedConv")

    runSerializationTest(conv, input)
  }

  "bigquant.Linear serializer" should "work properly " in {
    val outputSize = 2
    val inputSize = 2

    val kernelData = Array(
      2.0f, 3f,
      4f, 5f
    )

    val biasData = Array(0.0f, 0.1f)

    val input = Tensor[Float](2, 2).apply1(_ => Random.nextFloat())
    val weight = Tensor[Float](Storage(kernelData), 1, Array(outputSize, inputSize))
    val bias = Tensor[Float](Storage(biasData), 1, Array(outputSize))
    val linear = quantized.Linear[Float](outputSize, inputSize, initWeight = weight,
      initBias = bias).setName("quantLinear")
    runSerializationTest(linear, input)
  }

  // Below are TF Ops
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
    val assert = new Assert[Float]().setName("assert")
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
    val broadcastGradientArgs = BroadcastGradientArgs[Float]().
      setName("broadcastGradientArgs")
    val input =
      T(
        Tensor[Int](T(1, 2, 3)),
        Tensor[Int](T(2, 2, 1))
      )
    runSerializationTest(broadcastGradientArgs, input, broadcastGradientArgs.
      asInstanceOf[ModuleToOperation[Float]].module.getClass)
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

  "NoOp serializer" should "work properly" in {
    val noOp = NoOp[Float]().setName("noOp")
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

    val exampleParser = new ParseExample[Float](3, Seq(FloatType, LongType, StringType),
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

  // nn.tf package

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

  "SplitAndSelect serializer" should "work properly" in {
    val splitAndSelect = SplitAndSelect[Float](2, 1, 2).setName("splitSelect")
    val input = Tensor[Float](1, 6, 2).apply1(_ => Random.nextFloat())
    runSerializationTest(splitAndSelect, input)
  }

  "StrideSlice serialier" should "work properly" in {
    val strideSlice = new StrideSlice[Float](Array((1, 1, 2, 1))).setName("strideSlice")
    val input = Tensor[Float](2, 2, 2).apply1(_ => Random.nextFloat())
    runSerializationTest(strideSlice, input)
  }

  "Variable serializer" should "work properly" in {
    val out = Tensor[Float](2, 2, 2).apply1(_ => Random.nextFloat())
    val grad = Tensor[Float](2, 2, 2).apply1(_ => Random.nextFloat())
    val variable = Variable[Float](out, grad).setName("variable")
    val input = Tensor[Float](2, 2, 2).apply1(_ => Random.nextFloat())
    runSerializationTest(variable, input)
  }

  // tf.loaders

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
    val strideSliceLoadTF = new StridedSliceLoadTF[Float]().
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

  "Proposal serializer" should "work properly" in {
    val proposal = Proposal(200, 100, Array[Float](0.1f, 0.2f, 0.3f), Array[Float](4, 5, 6))
    val score = Tensor[Float](1, 18, 20, 30).randn()
    val boxes = Tensor[Float](1, 36, 20, 30).randn()
    val imInfo = Tensor[Float](T(300, 300, 1, 1)).resize(1, 4)
    val input = T(score, boxes, imInfo)
    runSerializationTest(proposal, input)
  }

  "NormalizeScale serializer" should "work properly" in {
    val module = NormalizeScale[Float](2, scale = 20, size = Array(1, 5, 1, 1),
      wRegularizer = L2Regularizer[Float](0.2)).setName("NormalizeScale")

    val input = Tensor[Float](1, 5, 3, 4).randn()
    runSerializationTest(module, input)
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

  "DetectionOutputSSD serializer" should "work properly" in {
    val module = DetectionOutputSSD[Float](DetectionOutputParam()).setName("DetectionOutputSSD")
    val name = module.getName
    val serFile = File.createTempFile(name, postFix)

    ModulePersister.saveToFile[Float](serFile.getAbsolutePath, null, module.evaluate(), true)
    RNG.setSeed(1000)
    val loadedModule = ModuleLoader.loadFromFile[Float](serFile.getAbsolutePath)


    if (serFile.exists) {
      serFile.delete
    }
    tested.add(module.getClass.getName)
  }

  "DetectionOutputFrcnn serializer" should "work properly" in {
    val module = DetectionOutputFrcnn().setName("DetectionOutputFrcnn")
    val name = module.getName
    val serFile = File.createTempFile(name, postFix)

    ModulePersister.saveToFile[Float](serFile.getAbsolutePath, null, module.evaluate(), true)
    RNG.setSeed(1000)
    val loadedModule = ModuleLoader.loadFromFile[Float](serFile.getAbsolutePath)


    if (serFile.exists) {
      serFile.delete
    }
    tested.add(module.getClass.getName)
  }

  "PriorBox serializer" should "work properly" in {
    val isClip = false
    val isFlip = true
    val variances = Array(0.1f, 0.1f, 0.2f, 0.2f)
    val minSizes = Array(460.8f)
    val maxSizes = Array(537.6f)
    val aspectRatios = Array(2f)
    val module = PriorBox[Float](minSizes = minSizes, maxSizes = maxSizes,
      _aspectRatios = aspectRatios, isFlip = isFlip, isClip = isClip,
      variances = variances, step = 0, offset = 0.5f, imgH = 512, imgW = 512)
    val input = Tensor[Float](8, 256, 1, 1)
    runSerializationTest(module, input)
  }

  override protected def afterAll() = {
    var total = 0
    expected.foreach(exp => {
      require(tested.contains(exp), s" $exp not included in the test!")
      total += 1
    })
    println(s"total $total, remaining ${expected.size - total}")
  }

}

