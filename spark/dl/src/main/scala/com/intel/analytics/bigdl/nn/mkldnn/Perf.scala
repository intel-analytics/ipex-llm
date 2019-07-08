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

package com.intel.analytics.bigdl.nn.mkldnn

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.mkl.{MKL, Memory, MklDnn}
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.mkldnn.Phase.{InferencePhase, TrainingPhase}
import com.intel.analytics.bigdl.nn.mkldnn.ResNet.DatasetType.ImageNet
import com.intel.analytics.bigdl.nn.mkldnn.models.Vgg_16
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{Engine, T, Table, ThreadPool}
import org.apache.log4j.Logger
import scopt.OptionParser

import scala.reflect.ClassTag

object Perf {

  val logger = Logger.getLogger(getClass)

  val parser = new OptionParser[ResNet50PerfParams]("BigDL w/ Dnn Local Model Performance Test") {
    opt[String]('m', "model")
      .text("model you want, vgg16 | resnet50 | vgg16_graph | resnet50_graph")
      .action((v, p) => p.copy(model = v))
    opt[Int]('b', "batchSize")
      .text("Batch size of input data")
      .action((v, p) => p.copy(batchSize = v))
    opt[Int]('i', "iteration")
      .text("Iteration of perf test. The result will be average of each iteration time cost")
      .action((v, p) => p.copy(iteration = v))
    opt[Boolean]('t', "training")
      .text(s"Perf test training or testing")
      .action((v, p) => p.copy(training = v))
  }

  def main(argv: Array[String]): Unit = {
    System.setProperty("bigdl.mkldnn.fusion.convbn", "true")
    System.setProperty("bigdl.mkldnn.fusion.bnrelu", "true")
    System.setProperty("bigdl.mkldnn.fusion.convrelu", "true")
    System.setProperty("bigdl.mkldnn.fusion.convsum", "true")

    System.setProperty("bigdl.localMode", "true")
    System.setProperty("bigdl.engineType", "mkldnn")
    Engine.init

    parser.parse(argv, new ResNet50PerfParams()).foreach { params =>
      val batchSize = params.batchSize
      val training = params.training
      val iterations = params.iteration

      val classNum = 1000

      val inputFormat = Memory.Format.nchw
      val inputShape = Array(batchSize, 3, 224, 224)
      val input = Tensor(inputShape).rand()
      val label = Tensor(batchSize).apply1(_ => Math.ceil(RNG.uniform(0, 1) * 1000).toFloat)

      val model = params.model match {
        case "vgg16" => Vgg_16(batchSize, classNum, true)
        case "resnet50" => ResNet(batchSize, classNum, T("depth" -> 50, "dataSet" -> ImageNet))
        case "vgg16_graph" => Vgg_16.graph(batchSize, classNum, true)
        case "resnet50_graph" =>
          ResNet.graph(batchSize, classNum, T("depth" -> 50, "dataSet" -> ImageNet))
        case _ => throw new UnsupportedOperationException(s"Unkown model ${params.model}")
      }

      val criterion = CrossEntropyCriterion()

      Engine.dnnComputing.invokeAndWait2(Array(1).map(_ => () => {
        if (training) {
          model.training()
          if (model.isInstanceOf[MklDnnContainer]) {
            model.asInstanceOf[MklDnnContainer]
              .compile(TrainingPhase, Array(HeapData(inputShape, inputFormat)))
          } else if (model.isInstanceOf[DnnGraph]) {
            model.asInstanceOf[DnnGraph].compile(TrainingPhase)
          }
        } else {
          model.evaluate()
          if (model.isInstanceOf[MklDnnContainer]) {
            model.asInstanceOf[MklDnnContainer]
              .compile(InferencePhase, Array(HeapData(inputShape, inputFormat)))
          } else if (model.isInstanceOf[DnnGraph]) {
            model.asInstanceOf[DnnGraph].compile(InferencePhase)
          }
        }
      }))

      var iteration = 0
      while (iteration < iterations) {
        val start = System.nanoTime()

        Engine.dnnComputing.invokeAndWait2(Array(1).map(_ => () => {
          val output = model.forward(input)

          if (training) {
            val _loss = criterion.forward(output, label)
            val errors = criterion.backward(output, label).toTensor
            model.backward(input, errors)
          }
        }))

        val takes = System.nanoTime() - start

        val throughput = "%.2f".format(batchSize.toFloat / (takes / 1e9))
        logger.info(s"Iteration $iteration, takes $takes s, throughput is $throughput imgs/sec")

        iteration += 1
      }
    }
  }
}

case class ResNet50PerfParams (
  batchSize: Int = 16,
  iteration: Int = 50,
  training: Boolean = true,
  model: String = "vgg16"
)

object ResNet {
  def modelInit(model: Module[Float]): Unit = {
    def initModules(model: Module[Float]): Unit = {
      model match {
        case container: Container[Activity, Activity, Float] =>
          container.modules.foreach(m => initModules(m))

        case conv: SpatialConvolution =>
          val n: Float = conv.kernelW * conv.kernelW * conv.nOutputPlane
          val weight = Tensor[Float].resize(conv.weight.size()).apply1 { _ =>
            RNG.normal(0, Math.sqrt(2.0f / n)).toFloat
          }
          val bias = Tensor[Float].resize(conv.bias.size()).apply1(_ => 0.0f)
          conv.weight.copy(weight)
          conv.bias.copy(bias)

        case bn: SpatialBatchNormalization =>
          val weightAndBias = Tensor[Float]().resize(Array(2, bn.nOutput))
          weightAndBias.select(1, 1).fill(1)
          weightAndBias.select(1, 2).fill(0)
          bn.weightAndBias.copy(weightAndBias.view(Array(bn.nOutput * 2)))

        case linear: Linear =>
          val bias = Tensor[Float](linear.bias.size()).apply1(_ => 0.0f)
          linear.bias.copy(bias)

        case _ =>
      }
    }

    initModules(model)
  }

  var iChannels = 0
  def apply(batchSize: Int, classNum: Int, opt: Table): Sequential = {

    val depth = opt.get("depth").getOrElse(18)
    val shortCutType = opt.get("shortcutType")
    val shortcutType = shortCutType.getOrElse(ShortcutType.B).asInstanceOf[ShortcutType]
    val dataSet = opt.getOrElse[DatasetType]("dataSet", DatasetType.CIFAR10)
    val optnet = opt.get("optnet").getOrElse(true)

    def shortcut(nInputPlane: Int, nOutputPlane: Int, stride: Int, name: String): Module[Float] = {
      val useConv = shortcutType == ShortcutType.C ||
        (shortcutType == ShortcutType.B && nInputPlane != nOutputPlane)

      if (useConv) {
        Sequential()
          .add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride, optnet = optnet)
            .setName(s"res${name}_branch1"))
          .add(SbnDnn(nOutputPlane).setName(s"bn${name}_branch1"))
      } else if (nInputPlane != nOutputPlane) {
        throw new IllegalArgumentException(s"useConv false")
      } else {
        Identity()
      }
    }

    def bottleneck(n: Int, stride: Int, name: String = ""): Module[Float] = {
      val nInputPlane = iChannels
      iChannels = n * 4

      val s = Sequential()
      s.add(Convolution(nInputPlane, n, 1, 1, 1, 1, 0, 0, optnet = optnet).setName(
          s"res${name}_branch2a"))
        .add(SbnDnn(n).setName(s"bn${name}_branch2a"))
        .add(ReLU().setName(s"res${name}_branch2a_relu"))
        .add(Convolution(n, n, 3, 3, stride, stride, 1, 1, optnet = optnet).setName(
          s"res${name}_branch2b"))
        .add(SbnDnn(n).setName(s"bn${name}_branch2b"))
        .add(ReLU().setName(s"res${name}_branch2b_relu"))
        .add(Convolution(n, n*4, 1, 1, 1, 1, 0, 0, optnet = optnet).setName(
          s"res${name}_branch2c"))
        .add(SbnDnn(n * 4).setInitMethod(Zeros, Zeros).setName(s"bn${name}_branch2c"))

      val model = Sequential()
        .add(ConcatTable().
          add(s).
          add(shortcut(nInputPlane, n*4, stride, name)).setName(s"$name/concatTable"))
        .add(CAddTable().setName(s"res$name"))
        .add(ReLU().setName(s"res${name}_relu"))
      model
    }

    def getName(i: Int, name: String): String = {
      val name1 = i match {
        case 1 => name + "a"
        case 2 => name + "b"
        case 3 => name + "c"
        case 4 => name + "d"
        case 5 => name + "e"
        case 6 => name + "f"
      }
      return name1
    }

    def layer(block: (Int, Int, String) => Module[Float], features: Int,
      count: Int, stride: Int = 1, name : String): Module[Float] = {
      val s = Sequential()
      for (i <- 1 to count) {
        s.add(block(features, if (i == 1) stride else 1, getName(i, name)))
      }
      s
    }

    val model = Sequential()
    if (dataSet == DatasetType.ImageNet) {
      val cfg = Map(
        50 -> ((3, 4, 6, 3), 2048, bottleneck: (Int, Int, String) => Module[Float])
      )

      require(cfg.keySet.contains(depth), s"Invalid depth ${depth}")

      val (loopConfig, nFeatures, block) = cfg.get(depth).get
      iChannels = 64

      model.add(Input(Array(batchSize, 3, 224, 224), Memory.Format.nchw))
        .add(SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3, propagateBack = false).setName("conv1"))
        .add(SbnDnn(64).setName("bn_conv1"))
        .add(ReLU().setName("conv1_relu"))
        .add(MaxPooling(3, 3, 2, 2).setName("pool1"))
        .add(layer(block, 64, loopConfig._1, name = "2"))
        .add(layer(block, 128, loopConfig._2, 2, name = "3"))
        .add(layer(block, 256, loopConfig._3, 2, name = "4"))
        .add(layer(block, 512, loopConfig._4, 2, name = "5"))
        .add(AvgPooling(7, 7, 1, 1).setName("pool5"))
        .add(Linear(nFeatures, classNum).setInitMethod(RandomNormal(0.0, 0.01), Zeros).setName(
          "fc1000"))
        .add(ReorderMemory(HeapData(Array(batchSize, classNum), Memory.Format.nc)))
    } else {
      throw new IllegalArgumentException(s"Invalid dataset ${dataSet}")
    }

    modelInit(model)
    model
  }

  def graph(batchSize: Int, classNum: Int, opt: Table): DnnGraph = {

    def modelInit(graph: DnnGraph): Unit = {
      graph.getSortedForwardExecutions.foreach(n => {
        n.element match {
          case conv: SpatialConvolution =>
            val n: Float = conv.kernelW * conv.kernelW * conv.nOutputPlane
            val weight = Tensor[Float].resize(conv.weight.size()).apply1 { _ =>
              RNG.normal(0, Math.sqrt(2.0f / n)).toFloat
            }
            val bias = Tensor[Float].resize(conv.bias.size()).apply1(_ => 0.0f)
            conv.weight.copy(weight)
            conv.bias.copy(bias)

          case bn: SpatialBatchNormalization =>
            val weightAndBias = Tensor[Float]().resize(Array(2, bn.nOutput))
            weightAndBias.select(1, 1).fill(1)
            weightAndBias.select(1, 2).fill(0)
            bn.weightAndBias.copy(weightAndBias.view(Array(bn.nOutput * 2)))

          case linear: Linear =>
            val bias = Tensor[Float](linear.bias.size()).apply1(_ => 0.0f)
            linear.bias.copy(bias)

          case _ =>
        }
      })
    }

    val depth = opt.get("depth").getOrElse(18)
    val shortCutType = opt.get("shortcutType")
    val shortcutType = shortCutType.getOrElse(ShortcutType.B).asInstanceOf[ShortcutType]
    val dataSet = opt.getOrElse[DatasetType]("dataSet", DatasetType.CIFAR10)
    val optnet = opt.get("optnet").getOrElse(true)

    def shortcut(input: ModuleNode[Float], nInputPlane: Int, nOutputPlane: Int,
                 stride: Int, name: String): ModuleNode[Float] = {
      val useConv = shortcutType == ShortcutType.C ||
        (shortcutType == ShortcutType.B && nInputPlane != nOutputPlane)

      if (useConv) {
        val conv = Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride, optnet = optnet)
            .setName(s"res${name}_branch1").inputs(input)
        SbnDnn(nOutputPlane).setName(s"bn${name}_branch1").inputs(conv)
      } else if (nInputPlane != nOutputPlane) {
        throw new IllegalArgumentException(s"useConv false")
      } else {
        Identity().inputs(input)
      }
    }

    def bottleneck(input: ModuleNode[Float], n: Int, stride: Int, name: String = "")
      : ModuleNode[Float] = {
      val nInputPlane = iChannels
      iChannels = n * 4

      val conv1 = Convolution(nInputPlane, n, 1, 1, 1, 1, 0, 0, optnet = optnet)
          .setName(s"res${name}_branch2a").inputs(input)
      val bn1 = SbnDnn(n).setName(s"bn${name}_branch2a").inputs(conv1)
      val relu1 = ReLU().setName(s"res${name}_branch2a_relu").inputs(bn1)
      val conv2 = Convolution(n, n, 3, 3, stride, stride, 1, 1, optnet = optnet).setName(
          s"res${name}_branch2b").inputs(relu1)
      val bn2 = SbnDnn(n).setName(s"bn${name}_branch2b").inputs(conv2)
      val relu3 = ReLU().setName(s"res${name}_branch2b_relu").inputs(bn2)
      val conv3 = Convolution(n, n*4, 1, 1, 1, 1, 0, 0, optnet = optnet).setName(
          s"res${name}_branch2c").inputs(relu3)
      val bn3 = SbnDnn(n * 4).setInitMethod(Zeros, Zeros).setName(
          s"bn${name}_branch2c").inputs(conv3)

      val short = shortcut(input, nInputPlane, n*4, stride, name)
      val cadd = CAddTable().setName(s"res$name").
          inputs(Array(bn3.asInstanceOf[ModuleNode[Float]], short))
      val relu = ReLU().setName(s"res${name}_relu").inputs(cadd)
      relu
    }

    def getName(i: Int, name: String): String = {
      val name1 = i match {
        case 1 => name + "a"
        case 2 => name + "b"
        case 3 => name + "c"
        case 4 => name + "d"
        case 5 => name + "e"
        case 6 => name + "f"
      }
      return name1
    }

    def layer(input: ModuleNode[Float],
              block: (ModuleNode[Float], Int, Int, String) => ModuleNode[Float],
              features: Int,
              count: Int, stride: Int = 1, name : String): ModuleNode[Float] = {
      var in = input
      for (i <- 1 to count) {
        val res = block(in, features, if (i == 1) stride else 1, getName(i, name))
        in = res
      }
      in
    }

    if (dataSet == DatasetType.ImageNet) {
      val cfg = Map(
        50 -> ((3, 4, 6, 3), 2048,
          bottleneck: (ModuleNode[Float], Int, Int, String) => ModuleNode[Float])
      )

      require(cfg.keySet.contains(depth), s"Invalid depth ${depth}")

      val (loopConfig, nFeatures, block) = cfg.get(depth).get
      iChannels = 64

      val input = Input(Array(batchSize, 3, 224, 224), Memory.Format.nchw).inputs()
      val conv1 = SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3, propagateBack = false)
        .setName("conv1").inputs(input)
      val bn1 = SbnDnn(64).setName("bn_conv1").inputs(conv1)
      val relu1 = ReLU().setName("conv1_relu").inputs(bn1)
      val pool1 = MaxPooling(3, 3, 2, 2).setName("pool1").inputs(relu1)
      val layer1 = layer(pool1, block, 64, loopConfig._1, name = "2")
      val layer2 = layer(layer1, block, 128, loopConfig._2, 2, name = "3")
      val layer3 = layer(layer2, block, 256, loopConfig._3, 2, name = "4")
      val layer4 = layer(layer3, block, 512, loopConfig._4, 2, name = "5")
      val pool2 = AvgPooling(7, 7, 1, 1).setName("pool5").inputs(layer4)
      val fc = Linear(nFeatures, classNum).setInitMethod(RandomNormal(0.0, 0.01), Zeros).setName(
          "fc1000").inputs(pool2)
      val output = ReorderMemory(HeapData(Array(batchSize, classNum), Memory.Format.nc)).inputs(fc)

      val model = DnnGraph(Array(input), Array(output))
      modelInit(model)
      model
    } else {
      throw new IllegalArgumentException(s"Invalid dataset ${dataSet}")
    }
  }

  /**
   * dataset type
   * @param typeId type id
   */
  sealed abstract class DatasetType(typeId: Int)
    extends Serializable

  /**
   *  define some dataset type
   */
  object DatasetType {
    case object CIFAR10 extends DatasetType(0)
    case object ImageNet extends DatasetType(1)
  }

  /**
   * ShortcutType
   * @param typeId type id
   */
  sealed abstract class ShortcutType(typeId: Int)
    extends Serializable

  /**
   * ShortcutType-A is used for Cifar-10, ShortcutType-B is used for ImageNet.
   * ShortcutType-C is used for others.
   */
  object ShortcutType{
    case object A extends ShortcutType(0)
    case object B extends ShortcutType(1)
    case object C extends ShortcutType(2)
  }
}

object Convolution {
  def apply(
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
    optnet: Boolean = true,
    weightDecay: Double = 1e-4): SpatialConvolution = {
    val conv = SpatialConvolution(nInputPlane, nOutputPlane, kernelW, kernelH,
      strideW, strideH, padW, padH, nGroup, propagateBack)
    conv.setInitMethod(MsraFiller(false), Zeros)
    conv
  }
}

object SbnDnn {
  def apply[@specialized(Float, Double) T: ClassTag](
    nOutput: Int,
    eps: Double = 1e-3,
    momentum: Double = 0.9)
    (implicit ev: TensorNumeric[T]): SpatialBatchNormalization = {
    SpatialBatchNormalization(nOutput, eps, momentum).setInitMethod(Ones, Zeros)
  }
}
