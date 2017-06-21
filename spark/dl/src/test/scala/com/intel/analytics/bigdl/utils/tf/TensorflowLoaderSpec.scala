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
package com.intel.analytics.bigdl.utils.tf

import java.io.{File => JFile}
import java.nio.ByteOrder
import java.util.UUID

import com.intel.analytics.bigdl.dataset.{DistributedDataSet, MiniBatch}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.{DistriOptimizer, Trigger}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import com.intel.analytics.bigdl.numeric.NumericFloat
import org.tensorflow.framework.NodeDef

import scala.collection.mutable
import scala.sys.process._
import scala.math._
import scala.reflect.ClassTag

object TensorflowLoaderSpec {
  private val data1 = Array(0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.1f)
  private val data2 = Array(0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.1f)
  private val input1: Tensor[Float] = Tensor[Float](Storage[Float](data1))
  private val input2: Tensor[Float] = Tensor[Float](Storage[Float](data2))
  private val nodeNumber = 4
  private val coreNumber = 4

  Engine.init(nodeNumber, coreNumber, true)

  private val batchSize = 2 * coreNumber

  private val prepareData: Int => (MiniBatch[Float]) = index => {
    val input = Tensor[Float]().resize(batchSize, 10)
    val target = Tensor[Float]().resize(batchSize)
    var i = 0
    while (i < batchSize) {
      if (i % 2 == 0) {
        target.setValue(i + 1, 0.0f)
        input.select(1, i + 1).copy(input1)
      } else {
        target.setValue(i + 1, 0.1f)
        input.select(1, i + 1).copy(input2)
      }
      i += 1
    }
    MiniBatch(input, target)
  }
}

class TensorflowLoaderSpec extends TensorflowSpecHelper{

  Logger.getLogger("org").setLevel(Level.WARN)
  Logger.getLogger("akka").setLevel(Level.WARN)

  import TensorflowLoaderSpec._

  var sc: SparkContext = null

  var dataSet: DistributedDataSet[MiniBatch[Float]] = null

  before {
    sc = new SparkContext("local[1]", "RDDOptimizerSpec")

    val rdd = sc.parallelize(1 to (256 * 4), 4).map(prepareData)

    dataSet = new DistributedDataSet[MiniBatch[Float]] {
      override def originRDD(): RDD[_] = rdd

      override def data(train : Boolean): RDD[MiniBatch[Float]] = rdd

      override def size(): Long = 256 * nodeNumber

      override def shuffle(): Unit = {}
    }

    Engine.model.setPoolSize(1)

    System.setProperty("bigdl.enableNHWC", "true")
  }

  after {
    if (sc != null) {
      sc.stop()
    }
    System.setProperty("bigdl.enableNHWC", "false")
  }

  "TensorFlow loader" should "read a list of nodes from pb file" in {
    val resource = getClass().getClassLoader().getResource("tf")
    val path = processPath(resource.getPath()) + JFile.separator + "test.pb"
    val results = TensorflowLoader.parse(path)
    results.size() should be(14)
  }

  "TensorFlow loader" should "be able to build a TF graph" in {
    val resource = getClass().getClassLoader().getResource("tf")
    val path = processPath(resource.getPath()) + JFile.separator + "test.pb"
    val results = TensorflowLoader.parse(path)
    val tfGraph = TensorflowLoader.buildTFGraph(results, Seq("output"))
    tfGraph.size should be(15)  // there's a dummy output
    val topSort = tfGraph.topologySort// It can do topology sort
    topSort.length should be(15)
    topSort(0).element should be(null)
    topSort(1).element.getName should be("output")
    topSort(2).element.getName should be("MatMul_1")
    topSort(3).element.getName should be("Variable_3/read")
    topSort(4).element.getName should be("Variable_3")
    topSort(5).element.getName should be("Tanh")
    topSort(6).element.getName should be("Variable_2/read")
    topSort(7).element.getName should be("Variable_2")
    topSort(8).element.getName should be("BiasAdd")
    topSort(9).element.getName should be("MatMul")
    topSort(10).element.getName should be("Variable_1/read")
    topSort(11).element.getName should be("Variable_1")
    topSort(12).element.getName should be("Placeholder")
    topSort(13).element.getName should be("Variable/read")
    topSort(14).element.getName should be("Variable")
  }

  "TensorFlow loader" should "be able to build a BigDL graph" in {
    val resource = getClass().getClassLoader().getResource("tf")
    val path = processPath(resource.getPath()) + JFile.separator + "test.pb"
    val model = TensorflowLoader.load(path, Seq("Placeholder"), Seq("output"),
      ByteOrder.LITTLE_ENDIAN)
    val container = model.asInstanceOf[Graph[Float]]
    container.modules.length should be(4)
    RandomGenerator.RNG.setSeed(100)
    val input = Tensor[Float](4, 1).rand()
    val output1 = container.forward(input)

    val model2 = Sequential[Float]()
    val fc1 = Linear[Float](1, 10)
    fc1.parameters()._1(0).fill(0.2f)
    fc1.parameters()._1(1).fill(0.1f)
    model2.add(fc1).add(Tanh())

    val fc2 = Linear[Float](10, 1)
    fc2.parameters()._1(0).fill(0.2f)
    fc2.parameters()._1(1).fill(0.1f)
    model2.add(fc2)

    val output2 = model2.forward(input)
    output1 should be(output2)
  }

  "Shared weights" should "be the same instance" in {
    tfCheck()
    val modelName = "share_weight"
    // Generate command and prepare the temp folder
    val s = JFile.separator
    val modelsFolder = processPath(getClass().getClassLoader().getResource("tf").getPath()) +
      s + "models"
    val modelScript = modelsFolder + s + s"$modelName.py"
    val tmpLocation = java.io.File.createTempFile("tensorflowLoaderTest" + UUID.randomUUID(),
      modelName)
    tmpLocation.delete()
    tmpLocation.mkdir()

    require(runPython(s"$modelScript $tmpLocation"), "error when run the model script")

    // Load the model and input/output tensors
    val modelFile = tmpLocation + s + "model.pb"
    val model = TensorflowLoader.load(modelFile, Seq("Placeholder"), Seq("output"),
      ByteOrder.LITTLE_ENDIAN)
    val container = model.asInstanceOf[Graph[Float]]
    val l1 = container.modules(1).asInstanceOf[Linear[Float]]
    val l2 = container.modules(3).asInstanceOf[Linear[Float]]
    assert(l1.weight eq l2.weight)
    assert(l1.bias eq l2.bias)
  }

  "Shared weights" should "be the same after running optimizer" in {
    tfCheck()
    val modelName = "share_weight"
    // Generate command and prepare the temp folder
    val s = JFile.separator
    val modelsFolder = processPath(getClass().getClassLoader().getResource("tf").getPath()) +
      s + "models"
    val modelScript = modelsFolder + s + s"$modelName.py"
    val tmpLocation = java.io.File.createTempFile("tensorflowLoaderTest" + UUID.randomUUID(),
      modelName)
    tmpLocation.delete()
    tmpLocation.mkdir()

    require(runPython(s"$modelScript $tmpLocation"), "error when run the model script")

    // Load the model and input/output tensors
    val modelFile = tmpLocation + s + "model.pb"
    val model = TensorflowLoader.load(modelFile, Seq("Placeholder"), Seq("output"),
      ByteOrder.LITTLE_ENDIAN)
    val container = model.asInstanceOf[Graph[Float]]

    val optimizer = new DistriOptimizer[Float](container, dataSet, new MSECriterion[Float]())
      .setState(T("learningRate" -> 20.0))
      .setEndWhen(Trigger.maxEpoch(5))
    optimizer.optimize()

    val l1 = container.modules(1).asInstanceOf[Linear[Float]]
    val l2 = container.modules(3).asInstanceOf[Linear[Float]]
    assert(l1.weight == l2.weight)
    assert(l1.bias == l2.bias)
  }

  "static simple rnn " should "have the same inference result as tensorflow" in {
    System.setProperty("bigdl.enableNHWC", "false")
    tfCheck()
    val modelName = "rnn"
    // Generate command and prepare the temp folder
    val s = JFile.separator
    val modelsFolder = processPath(getClass().getClassLoader().getResource("tf").getPath()) +
      s + "models"
    val modelScript = modelsFolder + s + s"$modelName.py"
    val tmpLocation = java.io.File.createTempFile("tensorflowLoaderTest" + UUID.randomUUID(),
      modelName)
    tmpLocation.delete()
    tmpLocation.mkdir()

    require(runPython(s"$modelScript $tmpLocation"), "error when run the model script")

    // Load the model and input/output tensors
    val modelFile = tmpLocation + s + "model.pb"


    val results = TensorflowLoader.parse(modelFile)
    val tfGraph = TensorflowLoader.buildTFGraph(results, Seq("output"))
    val model = TensorflowLoader.buildBigDLModel(tfGraph, Seq("input"),
      Seq("output"),
      ByteOrder.LITTLE_ENDIAN)
    val input = TensorflowToBigDL.toTensor(results.get(0).getAttrMap.get("value").getTensor,
      ByteOrder.LITTLE_ENDIAN).contiguous()
    val tfResult = TensorflowToBigDL.toTensor(results.get(results.size()-1)
      .getAttrMap.get("value").getTensor, ByteOrder.LITTLE_ENDIAN)
    val bigDLResult = model.forward(input)
    tfResult.almostEqual(bigDLResult.toTensor, 1e-6)
  }

  "static lstm rnn " should "have the same inference result as tensorflow" in {
    tfCheck()
    System.setProperty("bigdl.enableNHWC", "false")
    val modelName = "rnn_lstm"
    // Generate command and prepare the temp folder
    val s = JFile.separator
    val modelsFolder = processPath(getClass().getClassLoader().getResource("tf").getPath()) +
      s + "models"
    val modelScript = modelsFolder + s + s"$modelName.py"
    val tmpLocation = java.io.File.createTempFile("tensorflowLoaderTest" + UUID.randomUUID(),
      modelName)
    tmpLocation.delete()
    tmpLocation.mkdir()

    require(runPython(s"$modelScript $tmpLocation"), "error when run the model script")

    // Load the model and input/output tensors
    val modelFile = tmpLocation + s + "model.pb"

    val results = TensorflowLoader.parse(modelFile)
    val tfGraph = TensorflowLoader.buildTFGraph(results.subList(0, results.size()-1), Seq("output"))
    val model = TensorflowLoader.buildBigDLModel(tfGraph, Seq("input"),
      Seq("output"),
      ByteOrder.LITTLE_ENDIAN)
    val input = TensorflowToBigDL.toTensor(results.get(0).getAttrMap.get("value").getTensor,
      ByteOrder.LITTLE_ENDIAN).contiguous()
    val tfResult = TensorflowToBigDL.toTensor(results.get(results.size()-1)
      .getAttrMap.get("value").getTensor, ByteOrder.LITTLE_ENDIAN)
    val bigDLResult = model.forward(input)
    tfResult.almostEqual(bigDLResult.toTensor, 1e-5)
  }

  "Tensorflow lenet" should "be load correctly" in {
    testModelForward("lenet", Seq("LeNet/pool2/MaxPool:0"), true).foreach {
      case(tf, bigdl) =>
        val transpose = bigdl.transpose(2, 3).transpose(3, 4)
        tf.almostEqual(transpose, 1e-6) should be(true)
    }
    testModelBackward("lenet", Seq("LeNet/pool2/MaxPool:0"), true,
      Seq((4, 3), (3, 2))).foreach {
      case(tf, bigdl) =>
        if (tf.dim() == 4) {
          val trans = tf.transpose(1, 4).transpose(2, 3).transpose(3, 4).contiguous()
          trans.almostEqual(bigdl, 1e-4) should be(true)
        }
        else {
          tf.almostEqual(bigdl, 1e-4) should be(true)
        }
    }
  }

  "Tensorflow Alexnet" should "be load correctly" in {
    testModelForward("alexnet", Seq("alexnet_v2/fc8/squeezed:0"), true).foreach {
      case(tf, bigdl) =>
        tf.almostEqual(bigdl, 1e-7) should be(true)
    }
    testModelBackward("alexnet", Seq("alexnet_v2/fc8/squeezed:0"), true,
      Seq.empty).foreach {
      case(tf, bigdl) =>
        if (tf.dim() == 4) {
          val trans = tf.transpose(1, 4).transpose(2, 3).transpose(3, 4).contiguous()
          trans.almostEqual(bigdl, 1e-4) should be(true)
        }
        else {
          tf.almostEqual(bigdl, 1e-4) should be(true)
        }
    }
  }

  "TensorFlow vgg_a" should "be load correctly" in {
    testModelForward("vgga", Seq("vgg_a/fc8/squeezed:0"), true).foreach {
      case(tf, bigdl) =>
        tf.almostEqual(bigdl, 1e-7) should be(true)
    }
  }

  "TensorFlow vgg_16" should "be load correctly" in {
    testModelForward("vgg16", Seq("vgg_16/fc8/squeezed:0"), true).foreach {
      case(tf, bigdl) =>
        tf.almostEqual(bigdl, 1e-7) should be(true)
    }
  }

  "TensorFlow vgg_19" should "be load correctly" in {
    testModelForward("vgg19", Seq("vgg_19/fc8/squeezed:0"), true).foreach {
      case(tf, bigdl) =>
        tf.almostEqual(bigdl, 1e-7) should be(true)
    }
  }

  "TensorFlow overfeat" should "be load correctly" in {
    testModelForward("overfeat", Seq("overfeat/fc8/squeezed:0"), true).foreach {
      case(tf, bigdl) =>
        tf.almostEqual(bigdl, 1e-7) should be(true)
    }
  }

  "TensorFlow inception_v3" should "be load correctly" in {
    testModelForward("inception_v3", Seq("InceptionV3/Logits/SpatialSqueeze:0"), true).foreach {
      case(tf, bigdl) =>
        tf.almostEqual(bigdl, 1e-7) should be(true)
    }
  }

  "TensorFlow resnet_v1" should "be load correctly" in {
    testModelForward("resnet_v1", Seq("resnet_v1_101/SpatialSqueeze:0"), true).foreach {
      case(tf, bigdl) =>
        tf.almostEqual(bigdl, 1e-6) should be(true)
    }
  }

  "TensorFlow inception_resnet_v2" should "be load correctly" in {
    testModelForward("inception_resnet_v2", Seq("InceptionResnetV2/Logits/Logits/BiasAdd:0",
      "InceptionResnetV2/AuxLogits/Logits/BiasAdd:0"), true).foreach {
      case(tf, bigdl) =>
        tf.almostEqual(bigdl, 1e-7) should be(true)
    }
  }


  private def testModelForward(modelName: String, endPoints: Seq[String], transInput: Boolean)
  : Seq[(Tensor[Float], Tensor[Float])] = {

    tfCheck()
    // Generate command and prepare the temp folder
    val s = JFile.separator
    val modelsFolder = processPath(getClass().getClassLoader().getResource("tf").getPath()) +
      s + "models"
    val modelScript = modelsFolder + s + s"$modelName.py"
    val tmpLocation = java.io.File.createTempFile("tensorflowLoaderTest" + UUID.randomUUID(),
      modelName)
    tmpLocation.delete()
    tmpLocation.mkdir()

    require(runPython(s"$modelScript $tmpLocation ${endPoints.mkString(",")}"),
      "error when run the model script")

    // Load the model and input/output tensors
    import collection.JavaConverters._
    val modelFile = tmpLocation + s + "model.pb"
    val tfNodes = TensorflowLoader.parse(modelFile)

    // filter node for gradient computing
    val tfGraph = TensorflowLoader.buildTFGraph(tfNodes, endPoints.map(_.split(":")(0)))
    val context = new mutable.HashMap[NodeDef, (Tensor[Float], Tensor[Float])]
    val model = TensorflowLoader.buildBigDLModel(tfGraph, Seq("input"),
      endPoints.map(_.split(":")(0)), ByteOrder.LITTLE_ENDIAN, Some(context))

    // Compare the tensor contents
    val tfInputTensor = tfNodes.asScala.filter(_.getName == "input")(0)
      .getAttrMap.get("value").getTensor

    val tfOutputTensors = (0 until endPoints.length).map(
      i => tfNodes.asScala.filter(_.getName == s"output$i")(0).getAttrMap.get("value").getTensor)
    val input = TensorflowToBigDL.toTensor(tfInputTensor,
      ByteOrder.LITTLE_ENDIAN)

    val transposeInput = if (transInput) {
      input.transpose(2, 4).transpose(3, 4).contiguous()
    } else {
      input
    }

    val bigdlOutputs = if (endPoints.length == 1) {
      Seq(model.forward(transposeInput).toTensor)
    } else {
      val t = model.forward(transposeInput).toTable
      (1 to endPoints.length).map(t[Tensor[Float]](_))
    }

    val comparePair = tfOutputTensors.zip(bigdlOutputs).map{
      x =>
        val tensor = TensorflowToBigDL.toTensor(x._1, ByteOrder.LITTLE_ENDIAN)
        (tensor, x._2)
    }
    tmpLocation.deleteOnExit()
    comparePair
  }

  private def testModelBackward(
    modelName: String,
    endPoints: Seq[String],
    transInput: Boolean,
    transOutputSeq: Seq[(Int, Int)]): Seq[(Tensor[Float], Tensor[Float])] = {

    tfCheck()
    // Generate command and prepare the temp folder
    val s = JFile.separator
    val modelsFolder = processPath(getClass().getClassLoader().getResource("tf").getPath()) +
      s + "models"
    val modelScript = modelsFolder + s + s"$modelName.py"
    val tmpLocation = java.io.File.createTempFile("tensorflowLoaderTest" + UUID.randomUUID(),
      modelName)
    tmpLocation.delete()
    tmpLocation.mkdir()

    require(runPython(s"$modelScript $tmpLocation ${endPoints.mkString(",")}"),
      "error when run the model script")

    // Load the model and input/output tensors
    import collection.JavaConverters._
    val modelFile = tmpLocation + s + "model.pb"
    val tfNodes = TensorflowLoader.parse(modelFile)

    // filter node for gradient computing
    val tfGraph = TensorflowLoader.buildTFGraph(tfNodes, endPoints.map(_.split(":")(0)))
    val context = new mutable.HashMap[NodeDef, (Tensor[Float], Tensor[Float])]
    val model = TensorflowLoader.buildBigDLModel(tfGraph, Seq("input"),
      endPoints.map(_.split(":")(0)), ByteOrder.LITTLE_ENDIAN, Some(context))

    // Compare the tensor contents
    val tfInputTensor = tfNodes.asScala.filter(_.getName == "input")(0)
      .getAttrMap.get("value").getTensor

    val input = TensorflowToBigDL.toTensor(tfInputTensor,
      ByteOrder.LITTLE_ENDIAN)

    val transposeInput = if (transInput) {
      input.transpose(2, 4).transpose(3, 4).contiguous()
    } else {
      input
    }

    val bigdlOutputs = if (endPoints.length == 1) {
      Seq(model.forward(transposeInput).toTensor)
    } else {
      val t = model.forward(transposeInput).toTable
      (1 to endPoints.length).map(t[Tensor[Float]](_))
    }

    // get gradient input of tensorflow
    val gradInputs = (0 until endPoints.length).map{
      i =>
        val t = tfNodes.asScala.filter(_.getName == s"grad_input$i")(0)
          .getAttrMap.get("value").getTensor
        var tensor = TensorflowToBigDL.toTensor(t, ByteOrder.LITTLE_ENDIAN)
        for (trans <- transOutputSeq) {
          tensor = tensor.transpose(trans._1, trans._2)
        }
        tensor.contiguous()
    }

    // check shape equality here
    for (i <- 0 until endPoints.length) {
      bigdlOutputs(i).size() should be(gradInputs(i).size())
    }

    // find all gradients tensor in tensorflow graph
    val tfGradTensorsMap = context.keySet.map{
      node =>
        val t = tfNodes.asScala.filter(_.getName.contains(node.getName + "_grad"))(0)
        t.getName ->
          TensorflowToBigDL.toTensor(t.getAttrMap.get("value").getTensor, ByteOrder.LITTLE_ENDIAN)
    }.toMap


    val comparePair = new mutable.ArrayBuffer[(Tensor[Float], Tensor[Float])]()

    // do backward for each output and its corresponding gradient input
    for (i <- 0 until gradInputs.length) {
      // println(s"grad $i")
      model.backward(transposeInput, gradInputs(i))
      val pairs = context.keySet.map{
        x =>
          val name = s"${x.getName}_grad$i"
          // if (tfGradTensorsMap.contains(name)) {
          //   println(x.getName)
          //   context(x)._2.size().foreach(println(_))
          //   println(name)
          //   tfGradTensorsMap(name).size().foreach(println(_))
          // }
          (tfGradTensorsMap.get(name).getOrElse(null), context(x)._2)
      }.toSeq.filter(_._2 != null)
      comparePair ++= pairs
    }
    println(s"Compare ${comparePair.length} pairs of gradient vars in this graph")
    tmpLocation.deleteOnExit()
    comparePair
  }


  private def processPath(path: String): String = {
    if (path.contains(":")) {
      path.substring(1)
    } else {
      path
    }
  }
}
