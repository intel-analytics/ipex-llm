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

import breeze.linalg.Axis._1
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.example.languagemodel.PTBModel
import com.intel.analytics.bigdl.mkl.{AlgKind, Direction, Memory}
import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.models.resnet.ResNet.{DatasetType, ShortcutType}
import com.intel.analytics.bigdl.nn.mkldnn.Phase.{InferencePhase, TrainingPhase}
import com.intel.analytics.bigdl.nn.{Graph, Module => _, _}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils._
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import com.intel.analytics.bigdl.models.resnet
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.utils.intermediate._
import com.intel.analytics.bigdl.numeric.NumericFloat
import org.apache.spark.SparkContext

class DnnGraphSpec extends FlatSpec with Matchers with BeforeAndAfter {

  private var sc: SparkContext = _

  before {
    val nodeNumber = 1
    val coreNumber = 4
    Engine.init(nodeNumber, coreNumber, onSpark = true)
    sc = new SparkContext("local[1]", "DnnGraphSpec")
  }

  after {
    if (sc != null) {
      sc.stop()
    }
  }

  def model(size: Array[Int]) : Module[Float] = {
    val input = mkldnn.Input(size, Memory.Format.nchw).inputs()
    val conv1 = mkldnn.SpatialConvolution(1, 6, 5, 5).setName("conv1_5x5").inputs(input)
    val tanh1 = mkldnn.BlasWrapper(Tanh[Float]()).inputs(conv1)
    val pool1 = mkldnn.MaxPooling(2, 2, 2, 2).inputs(tanh1)
    val conv2 = BlasWrapper(
      nn.SpatialConvolution[Float](6, 12, 5, 5)).setName("conv2_5x5").inputs(pool1)
    val tanh2 = mkldnn.BlasWrapper(Tanh[Float]()).inputs(conv2)
    val pool2 = mkldnn.MaxPooling(2, 2, 2, 2).inputs(tanh2)
    val fc1 = mkldnn.Linear(12 * 4 * 4, 100).setName("fc1").inputs(pool2)
    val tanh3 = mkldnn.BlasWrapper(Tanh[Float]()).inputs(fc1)
    val fc2 = mkldnn.BlasWrapper(nn.Linear[Float](100, 10)).setName("fc2").inputs(tanh3)
    val output = mkldnn.BlasWrapper(LogSoftMax[Float]()).inputs(fc2)

    DnnGraph(Seq(input), Seq(output))
  }

  "Dnn vgg16 graph model" should "be correct" in {
    val batchSize = 2
    val seed = 1
    val inputFormat = Memory.Format.nchw
    val inputShape = Array(batchSize, 3, 224, 224)

    RNG.setSeed(seed)
    val graphModel = models.Vgg_16.graph(batchSize, 1000, false)
    RNG.setSeed(seed)
    val dnnModle = models.Vgg_16(batchSize, 1000, false)

    graphModel.asInstanceOf[DnnGraph].compile(TrainingPhase)
    dnnModle.compile(TrainingPhase)

    val input = Tensor[Float](inputShape).rand()
    val gradOutput = Tensor[Float](batchSize, 1000).rand()

    for (i <- 0 to 2) {
      graphModel.forward(input)
      dnnModle.forward(input)

      graphModel.backward(input, gradOutput)
      dnnModle.backward(input, gradOutput)
    }
    val output = Tools.dense(graphModel.forward(input)).toTensor[Float]
    val outputDnn = Tools.dense(dnnModle.forward(input)).toTensor[Float]

    val gradInput = Tools.dense(graphModel.backward(input, gradOutput)).toTensor[Float]
    val gradInputDnn = Tools.dense(dnnModle.backward(input, gradOutput)).toTensor[Float]

    output.almostEqual(outputDnn, 1e-4) should be(true)
    gradInput.almostEqual(gradInputDnn, 1e-4) should be (true)

    val p1 = dnnModle.getParameters()
    val p2 = graphModel.getParameters()
    p1._1.almostEqual(p2._1, 1e-4) should be(true)
    p1._2 almostEqual(p2._2, 1e-4) should be(true)
  }

  "Dnn Lenet graph model" should "be correct" in {
    val batchSize = 2
    val seed = 1
    val inputFormat = Memory.Format.nchw
    val inputShape = Array(batchSize, 1, 28, 28)

    RNG.setSeed(seed)
    val graphModel = LeNet5.dnnGraph(batchSize, 10)
    RNG.setSeed(seed)
    val dnnModle = LeNet5.dnn(batchSize, 10)

    graphModel.asInstanceOf[DnnGraph].compile(TrainingPhase)
    dnnModle.compile(TrainingPhase)

    val input = Tensor[Float](inputShape).rand()
    val gradOutput = Tensor[Float](batchSize, 10).rand()

    for (i <- 0 to 2) {
      graphModel.forward(input)
      dnnModle.forward(input)

      graphModel.backward(input, gradOutput)
      dnnModle.backward(input, gradOutput)
    }
    val output = Tools.dense(graphModel.forward(input)).toTensor[Float]
    val outputDnn = Tools.dense(dnnModle.forward(input)).toTensor[Float]

    val gradInput = Tools.dense(graphModel.backward(input, gradOutput)).toTensor[Float]
    val gradInputDnn = Tools.dense(dnnModle.backward(input, gradOutput)).toTensor[Float]

    output.almostEqual(outputDnn, 1e-4) should be(true)
    gradInput.almostEqual(gradInputDnn, 1e-4) should be (true)

    val p1 = dnnModle.getParameters()
    val p2 = graphModel.getParameters()
    p1._1.almostEqual(p2._1, 1e-4) should be(true)
    p1._2 almostEqual(p2._2, 1e-4) should be(true)
  }

  "ResNet50 graph model" should "be correct" in {
    val batchSize = 2
    val seed = 1
    val inputFormat = Memory.Format.nchw
    val inputShape = Array(batchSize, 3, 224, 224)

    RNG.setSeed(seed)
    val dnnModle = mkldnn.ResNet(batchSize, 1000, T("depth" -> 50,
      "dataSet" -> ResNet.DatasetType.ImageNet))
    RNG.setSeed(seed)
    val graphModel = mkldnn.ResNet.graph(batchSize, 1000, T("depth" -> 50,
      "dataSet" -> ResNet.DatasetType.ImageNet))

    val input = Tensor[Float](inputShape).rand()
    val gradOutput = Tensor[Float](batchSize, 1000).rand()

    graphModel.asInstanceOf[DnnGraph].compile(TrainingPhase)
    dnnModle.compile(TrainingPhase)

    for (i <- 0 to 2) {
      graphModel.forward(input)
      dnnModle.forward(input)

      graphModel.backward(input, gradOutput)
      dnnModle.backward(input, gradOutput)
    }
    val output = Tools.dense(graphModel.forward(input)).toTensor[Float]
    val outputDnn = Tools.dense(dnnModle.forward(input)).toTensor[Float]

    val gradInput = Tools.dense(graphModel.backward(input, gradOutput)).toTensor[Float]
    val gradInputDnn = Tools.dense(dnnModle.backward(input, gradOutput)).toTensor[Float]

    output.almostEqual(outputDnn, 1e-4) should be(true)
    gradInput.almostEqual(gradInputDnn, 1e-4) should be (true)

    val p1 = graphModel.getParametersTable()
    val p2 = dnnModle.getParametersTable()
    val keys = p1.keySet
    for (i <- keys) {
      val k = i.asInstanceOf[String]
      val t1 = p1[Table](k)
      val t2 = p2[Table](k)
      t1 should be(t2)
    }
  }

  "DnnGraph skip primitives" should "be correct" in {
    Engine.setEngineType(MklDnn)
    val batchSize = 2
    val inputShape = Array(batchSize, 1, 28, 28)
    val input = Tensor[Float](inputShape).rand()

    val dnn = model(inputShape).asInstanceOf[DnnGraph]
    dnn.evaluate()
    dnn.compile(Phase.InferencePhase)

    dnn.forward(input).toTensor[Float]
  }

  "Dnn graph fusion operation for resnet50" should "be correct" in {
    System.setProperty("bigdl.mkldnn.fusion.convbn", "true")
    System.setProperty("bigdl.mkldnn.fusion.bnrelu", "true")
    System.setProperty("bigdl.mkldnn.fusion.convrelu", "true")
    System.setProperty("bigdl.mkldnn.fusion.convsum", "true")
    System.setProperty("bigdl.mkldnn.fusion", "true")

    val batchSize = 2
    val seed = 1
    val inputFormat = Memory.Format.nchw
    val inputShape = Array(batchSize, 3, 224, 224)

    RNG.setSeed(seed)
    val seqModel = mkldnn.ResNet(batchSize, 1000, T("depth" -> 50,
      "dataSet" -> ResNet.DatasetType.ImageNet))
    RNG.setSeed(seed)
    val graphFuse = mkldnn.ResNet.graph(batchSize, 1000, T("depth" -> 50,
      "dataSet" -> ResNet.DatasetType.ImageNet))

    seqModel.getExtraParameter().map(_.fill(1.0f))
    graphFuse.getExtraParameter().map(_.fill(1.0f))

    seqModel.evaluate()
    seqModel.asInstanceOf[MklDnnContainer].compile(
      Phase.InferencePhase, Array(HeapData(inputShape, inputFormat)))
    graphFuse.evaluate()
    graphFuse.asInstanceOf[DnnGraph].compile(Phase.InferencePhase)

    RNG.setSeed(100)
    val input = Tensor[Float](inputShape).rand()

    val output = seqModel.forward(input).toTensor[Float]
    val outputFuse = graphFuse.forward(input).toTensor[Float]

    output.almostEqual(outputFuse, 1e-4) should be(true)

    System.clearProperty("bigdl.mkldnn.fusion.convbn")
    System.clearProperty("bigdl.mkldnn.fusion.bnrelu")
    System.clearProperty("bigdl.mkldnn.fusion.convrelu")
    System.clearProperty("bigdl.mkldnn.fusion.convsum")
    System.clearProperty("bigdl.mkldnn.fusion")
  }

  "Dnn graph fusion operation for vgg16" should "be correct" in {
    System.setProperty("bigdl.mkldnn.fusion.convbn", "true")
    System.setProperty("bigdl.mkldnn.fusion.bnrelu", "true")
    System.setProperty("bigdl.mkldnn.fusion.convrelu", "true")
    System.setProperty("bigdl.mkldnn.fusion.convsum", "true")
    System.setProperty("bigdl.mkldnn.fusion", "true")

    val batchSize = 2
    val seed = 1
    val inputFormat = Memory.Format.nchw
    val inputShape = Array(batchSize, 3, 224, 224)

    RNG.setSeed(seed)
    val seqModel = models.Vgg_16(batchSize, 1000, false)
    RNG.setSeed(seed)
    val graphFuse = models.Vgg_16.graph(batchSize, 1000, false)

    seqModel.evaluate()
    graphFuse.evaluate()
    graphFuse.asInstanceOf[DnnGraph].compile(Phase.InferencePhase)
    seqModel.compile(Phase.InferencePhase)

    val input = Tensor[Float](inputShape).rand()

    val output = Tools.dense(graphFuse.forward(input)).toTensor[Float]
    val outputDnn = Tools.dense(seqModel.forward(input)).toTensor[Float]

    output.almostEqual(outputDnn, 1e-4) should be(true)

    System.clearProperty("bigdl.mkldnn.fusion.convbn")
    System.clearProperty("bigdl.mkldnn.fusion.bnrelu")
    System.clearProperty("bigdl.mkldnn.fusion.convrelu")
    System.clearProperty("bigdl.mkldnn.fusion.convsum")
    System.clearProperty("bigdl.mkldnn.fusion")
  }

  "DnnGraph fusion" should "not change model parameters" in {
    Engine.setEngineType(MklDnn)
    import com.intel.analytics.bigdl.models.resnet
    RNG.setSeed(100)
    val module = resnet.ResNet(1000, T("shortcutType" -> ShortcutType.B, "depth" -> 50,
      "optnet" -> false, "dataSet" -> DatasetType.ImageNet))
      .toGraph().asInstanceOf[StaticGraph[Float]]
      .toIRgraph()

    val bcast = ModelBroadcast[Float]().broadcast(sc, module.evaluate())
    for(i <- 1 to 3) {
      val data = sc.parallelize(0 to 10, 1)
      data.mapPartitions(i => {
        val tensor = Tensor[Float](2, 3, 224, 224).rand()
        val mod = bcast.value()
        Iterator(mod.forward(tensor).toTensor[Float])
      }).count()

      sc.parallelize(1 to 1, 1).mapPartitions(i => {
        val weightSum = bcast.value().getWeightsBias().map(f => f.sum()).sum
        require(weightSum == 11759.763f, s"sum of model weight " +
          s"parameters should be 11759.764, but get ${weightSum}")
        i
      }).count()

      Engine.setEngineType(MklBlas)
    }
  }

  "DnnGraph with ntc" should "work correct" in {
    val vocabSize = 10001
    val hiddenSize = 256
    val numLayers = 1
    val batchSize = 8
    val seqLength = 16
    val inputSize = vocabSize
    val outputSize = vocabSize
    val f = AlgKind.EltwiseTanh
    val direction = Direction.UnidirectionalLeft2Right
    var i = 2

    val inputShape = Array[Int](batchSize, seqLength)
    val input = mkldnn.Input(inputShape, Memory.Format.nc).inputs()
    val embeddingLookup = BlasWrapper(LookupTable[Float](inputSize, hiddenSize)).inputs(input)
    val lstm = mkldnn.RNN(AlgKind.VanillaLstm, hiddenSize, hiddenSize, f = f, direction = direction)
      .inputs(embeddingLookup)
    val linear = BlasWrapper(TimeDistributed[Float](nn.Linear[Float](hiddenSize, outputSize)))
      .inputs(lstm)
    val output = mkldnn.Output(Memory.Format.ntc).inputs(linear)

    val dnn = DnnGraph(Array(input), Array(output))
    dnn.compile(Phase.TrainingPhase)

    val inputTensor = Tensor[Float](batchSize, seqLength).apply1(n => {
      i += 1
      i
    })
    val gradOutput = Tensor[Float](batchSize, seqLength, outputSize).rand()

    dnn.forward(inputTensor)
    dnn.backward(inputTensor, gradOutput)
  }
}
