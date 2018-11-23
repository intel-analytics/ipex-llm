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

import com.google.protobuf.ByteString
import com.intel.analytics.bigdl.dataset.{DistributedDataSet, MiniBatch}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.{DistriOptimizer, Trigger}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import com.intel.analytics.bigdl.numeric.NumericFloat
import org.tensorflow.framework.{DataType, NodeDef, TensorProto, TensorShapeProto}

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

  override def doBefore(): Unit = {
    val conf = Engine.createSparkConf().setAppName("RDDOptimizerSpec")
      .setMaster("local[1]")
    sc = SparkContext.getOrCreate(conf)

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

  override def doAfter(): Unit = {
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
    val (tfGraph, _, _) = TensorflowLoader.buildTFGraph(results, Seq("output"))
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

  "TensorFlow loader" should "be able to build a TF sub graph" in {
    val resource = getClass().getClassLoader().getResource("tf")
    val path = processPath(resource.getPath()) + JFile.separator + "test.pb"
    val results = TensorflowLoader.parse(path)
    val (tfGraph, _, _) = TensorflowLoader.buildTFGraph(results, Seq("output"),
      (node: NodeDef) => node.getName == "Tanh")
    tfGraph.size should be(9)  // there's a dummy output
    val topSort = tfGraph.topologySort// It can do topology sort
    topSort.length should be(9)
    topSort(0).element should be(null)
    topSort(1).element.getName should be("output")
    topSort(2).element.getName should be("MatMul_1")
    topSort(3).element.getName should be("Variable_3/read")
    topSort(4).element.getName should be("Variable_3")
    topSort(5).element.getName should be("Tanh")
    topSort(6).element.getName should be("Variable_2/read")
    topSort(7).element.getName should be("Variable_2")
    topSort(8).element.getName should be("input0")
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

  "TensorFlow loader" should "throw exception if input contain duplicate names" in {
    val resource = getClass().getClassLoader().getResource("tf")
    val path = processPath(resource.getPath()) + JFile.separator + "test.pb"
    intercept[IllegalArgumentException] {
      val model = TensorflowLoader.load(path, Seq("Placeholder", "Placeholder"), Seq("output"),
        ByteOrder.LITTLE_ENDIAN)
    }
  }

  "TensorFlow loader" should "throw exception if input contain conflict names" in {
    val resource = getClass().getClassLoader().getResource("tf")
    val path = processPath(resource.getPath()) + JFile.separator + "test.pb"
    intercept[IllegalArgumentException] {
      val model = TensorflowLoader.load(path, Seq("Placeholder", "Placeholder:0"), Seq("output"),
        ByteOrder.LITTLE_ENDIAN)
    }
  }

  "TensorFlow loader" should "throw exception if input location is incorrect" in {
    val resource = getClass().getClassLoader().getResource("tf")
    val path = processPath(resource.getPath()) + JFile.separator + "test.pb"
    intercept[IllegalArgumentException] {
      val model = TensorflowLoader.load(path, Seq("MatMul:2"), Seq("output"),
        ByteOrder.LITTLE_ENDIAN)
    }
  }

  "TensorFlow loader" should "be able to build a BigDL graph with specify input location" in {
    val resource = getClass().getClassLoader().getResource("tf")
    val path = processPath(resource.getPath()) + JFile.separator + "test.pb"
    val model = TensorflowLoader.load(path, Seq("MatMul:0"), Seq("output"),
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

  "TensorFlow loader" should "be able to build a BigDL graph from a subset of a tf graph" in {
    val resource = getClass().getClassLoader().getResource("tf")
    val path = processPath(resource.getPath()) + JFile.separator + "test.pb"
    val model = TensorflowLoader.load(path, Seq("Tanh"), Seq("output"),
      ByteOrder.LITTLE_ENDIAN)
    val container = model.asInstanceOf[Graph[Float]]
    container.modules.length should be(3)
    RandomGenerator.RNG.setSeed(100)
    val input = Tensor[Float](4, 10).rand()
    val output1 = container.forward(input)

    val model2 = Sequential[Float]()
    model2.add(Tanh())

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
      .setEndWhen(Trigger.maxEpoch(1))
    optimizer.optimize()

    val l1 = container.modules(1).asInstanceOf[Linear[Float]]
    val l2 = container.modules(3).asInstanceOf[Linear[Float]]
    assert(l1.weight == l2.weight)
    assert(l1.bias == l2.bias)
  }

  "static simple rnn " should "have the same result as tensorflow" in {
    val output = Seq("output:0")
    val comparePairs = testModel("rnn", output, backward = true)
    for (i <- output.indices) {
      val (tf, bigdl) = comparePairs(i)
      tf.almostEqual(bigdl, 1e-5) should be(true)
    }
    for (i <- output.length until comparePairs.length) {
      val (tf, bigdl) = comparePairs(i)
      tf.almostEqual(bigdl, 1e-3) should be(true)
    }
  }

  "static lstm rnn " should "have the same result as tensorflow" in {
    val output = Seq("output:0")
    val comparePairs = testModel("rnn_lstm", output, backward = true)
    for (i <- output.indices) {
      val (tf, bigdl) = comparePairs(i)
      tf.almostEqual(bigdl, 1e-5) should be(true)
    }
    for (i <- output.length until comparePairs.length) {
      val (tf, bigdl) = comparePairs(i)
      tf.almostEqual(bigdl, 1e-2) should be(true)
    }
  }

  "hand coded lstm rnn " should "have the same result as tensorflow" in {
    val output = Seq("output:0")
    val comparePairs = testModel("decoder", output, backward = true)
    for (i <- output.indices) {
      val (tf, bigdl) = comparePairs(i)
      tf.almostEqual(bigdl, 1e-5) should be(true)
    }
    for (i <- output.length until comparePairs.length) {
      val (tf, bigdl) = comparePairs(i)
      tf.almostEqual(bigdl, 1e-2) should be(true)
    }
  }

  "TensorFlow control dep" should "be load correctly" in {
    val output = Seq("output:0")
    val comparePairs = testModel("control_dep", output, backward = true)
    for (i <- output.indices) {
      val (tf, bigdl) = comparePairs(i)
      tf.almostEqual(bigdl, 1e-7) should be(true)
    }
    for (i <- output.length until comparePairs.length) {
      val (tf, bigdl) = comparePairs(i)
      tf.almostEqual(bigdl, 1e-3) should be(true)
    }
  }

  "Tensorflow load " should "be able to handle multiple edges" in {
    val output = Seq("output:0")
    val comparePairs = testModel("two_edge", output, backward = true)
    for (i <- output.indices) {
      val (tf, bigdl) = comparePairs(i)
      tf.almostEqual(bigdl, 1e-7) should be(true)
    }
    for (i <- output.length until comparePairs.length) {
      val (tf, bigdl) = comparePairs(i)
      tf.almostEqual(bigdl, 1e-3) should be(true)
    }
  }

  "Tensorflow batchnorm nhwc" should "be loaded correctly" in {
    val output = Seq("output:0")
    val comparePairs = testModel("batch_norm_nhwc", output, backward = true)
    for (i <- output.indices) {
      val (tf, bigdl) = comparePairs(i)
      tf.almostEqual(bigdl, 1e-4) should be(true)
    }
    for (i <- output.length until comparePairs.length) {
      val (tf, bigdl) = comparePairs(i)
      tf.almostEqual(bigdl, 1e-3) should be(true)
    }
  }

  "Tensorflow batchnorm nchw" should "be loaded correctly" in {
    val output = Seq("output:0")
    val comparePairs = testModel("batch_norm_nchw", output, backward = true)

    for (i <- output.indices) {
      val (tf, bigdl) = comparePairs(i)
      tf.almostEqual(bigdl, 1e-4) should be(true)
    }
    for (i <- output.length until comparePairs.length) {
      val (tf, bigdl) = comparePairs(i)
      tf.almostEqual(bigdl, 1e-3) should be(true)
    }
  }

  "Tensorflow lenet" should "be load correctly" in {
    val output = Seq("LeNet/pool2/MaxPool:0")
    val comparePairs = testModel("lenet", output, backward = true)

    for (i <- output.indices) {
      val (tf, bigdl) = comparePairs(i)
      tf.almostEqual(bigdl, 1e-6) should be(true)
    }
    for (i <- output.length until comparePairs.length) {
      val (tf, bigdl) = comparePairs(i)
      tf.almostEqual(bigdl, 1e-4) should be(true)
    }
  }

  "Tensorflow conv1d" should "be load correctly" in {
    val output = Seq("output:0")
    val comparePairs = testModel("temporal_convolution", output, backward = false)
    for (i <- output.indices) {
      val (tf, bigdl) = comparePairs(i)
      tf.almostEqual(bigdl, 1e-6) should be(true)
    }
  }

  "Tensorflow Alexnet" should "be load correctly" in {
    val output = Seq("alexnet_v2/fc8/squeezed:0")
    val comparePairs = testModel("alexnet", output, backward = true)
    for (i <- output.indices) {
      val (tf, bigdl) = comparePairs(i)
      tf.almostEqual(bigdl, 1e-6) should be(true)
    }
    for (i <- output.length until comparePairs.length) {
      val (tf, bigdl) = comparePairs(i)
      tf.almostEqual(bigdl, 1e-4) should be(true)
    }
  }

  "TensorFlow vgg_a" should "be load correctly" in {
    val output = Seq("vgg_a/fc8/squeezed:0")
    val comparePairs = testModel("vgga", output, backward = true)
    for (i <- output.indices) {
      val (tf, bigdl) = comparePairs(i)
      tf.almostEqual(bigdl, 1e-7) should be(true)
    }
    for (i <- output.length until comparePairs.length) {
      val (tf, bigdl) = comparePairs(i)
      tf.almostEqual(bigdl, 1e-3) should be(true)
    }
  }

  "TensorFlow vgg_16" should "be load correctly" in {
    val output = Seq("vgg_16/fc8/squeezed:0")
    val comparePairs = testModel("vgg16", output, backward = true)
    for (i <- output.indices) {
      val (tf, bigdl) = comparePairs(i)
      tf.almostEqual(bigdl, 1e-7) should be(true)
    }
    for (i <- output.length until comparePairs.length) {
      val (tf, bigdl) = comparePairs(i)
      tf.almostEqual(bigdl, 1e-2) should be(true)
    }
  }

  "TensorFlow vgg_19" should "be load correctly" in {
    val output = Seq("vgg_19/fc8/squeezed:0")
    val comparePairs = testModel("vgg19", output, backward = true)
    for (i <- output.indices) {
      val (tf, bigdl) = comparePairs(i)
      tf.almostEqual(bigdl, 1e-7) should be(true)
    }
    for (i <- output.length until comparePairs.length) {
      val (tf, bigdl) = comparePairs(i)
      tf.almostEqual(bigdl, 1e-2) should be(true)
    }
  }

  "TensorFlow overfeat" should "be load correctly" in {
    val output = Seq("overfeat/fc8/squeezed:0")
    val comparePairs = testModel("overfeat", output, backward = true)
    for (i <- output.indices) {
      val (tf, bigdl) = comparePairs(i)
      tf.almostEqual(bigdl, 1e-6) should be(true)
    }
    for (i <- output.length until comparePairs.length) {
      val (tf, bigdl) = comparePairs(i)
      tf.almostEqual(bigdl, 1e-3) should be(true)
    }
  }

  "TensorFlow inception_v3" should "be load correctly" in {
    val output = Seq("InceptionV3/Logits/SpatialSqueeze:0")
    val comparePairs = testModel("inception_v3", output, true)
    for (i <- output.indices) {
      val (tf, bigdl) = comparePairs(i)
      tf.almostEqual(bigdl, 1e-6) should be(true)
    }
    for (i <- output.length until comparePairs.length) {
      val (tf, bigdl) = comparePairs(i)
      tf.almostEqual(bigdl, 1e-2) should be(true)
    }
  }

  "TensorFlow resnet_v1" should "be load correctly" in {
    val output = Seq("resnet_v1_101/SpatialSqueeze:0")
    val comparePairs = testModel("resnet_v1", output, backward = true)
    for (i <- output.indices) {
      val (tf, bigdl) = comparePairs(i)
      tf.almostEqual(bigdl, 1e-6) should be(true)
    }
    for (i <- output.length until comparePairs.length) {
      val (tf, bigdl) = comparePairs(i)
      tf.almostEqual(bigdl, 1e-1) should be(true)
    }
  }

  "TensorFlow inception_resnet_v2" should "be load correctly" in {
    val output = Seq("InceptionResnetV2/Logits/Logits/BiasAdd:0",
      "InceptionResnetV2/AuxLogits/Logits/BiasAdd:0")
    val comparePairs = testModel("inception_resnet_v2", output, backward = true)
    for (i <- output.indices) {
      val (tf, bigdl) = comparePairs(i)
      tf.almostEqual(bigdl, 1e-6) should be(true)
    }
    for (i <- output.length until comparePairs.length) {
      val (tf, bigdl) = comparePairs(i)
      tf.almostEqual(bigdl, 1e-1) should be(true)
    }
  }

  "TensorArray operations" should "be load correctly" in {
    val output = Seq("scatter_and_gather:0", "split_and_concat:0", "write_and_read:0", "size1:0",
      "size2:0", "unstack_and_stack:0")
    val comparePairs = testModel("tensor_array", output, backward = false)
    for (i <- output.indices) {
      val (tf, bigdl) = comparePairs(i)
      tf.almostEqual(bigdl, 1e-6) should be(true)
    }
  }

  "dynamic rnn" should "be load correctly" in {
    val output = Seq("rnn_loss:0")
    val comparePairs = testModel("dynamic_rnn", output, backward = false)
    for (i <- output.indices) {
      val (tf, bigdl) = comparePairs(i)
      tf.almostEqual(bigdl, 1e-3) should be(true)
    }
  }

  "dynamic rnn grad" should "be load correctly" in {
    val output = Seq("gradOutput:0")
    val comparePairs = testModel("dynamic_rnn_grad", output, backward = false)
    for (i <- output.indices) {
      val (tf, bigdl) = comparePairs(i)
      tf.almostEqual(bigdl, 1e-3) should be(true)
    }
  }

  "dynamic lstm" should "be load correctly" in {
    val output = Seq("lstm_loss:0")
    val comparePairs = testModel("dynamic_lstm", output, backward = false)
    for (i <- output.indices) {
      val (tf, bigdl) = comparePairs(i)
      tf.almostEqual(bigdl, 1e-2) should be(true)
    }
  }

  "dynamic lstm grad" should "be load correctly" in {
    val output = Seq("gradOutput:0")
    val comparePairs = testModel("dynamic_lstm_grad", output, backward = false)
    for (i <- output.indices) {
      val (tf, bigdl) = comparePairs(i)
      tf.almostEqual(bigdl, 1e-2) should be(true)
    }
  }

  "dynamic gru" should "be load correctly" in {
    val output = Seq("gru_loss:0")
    val comparePairs = testModel("dynamic_gru", output, backward = false)
    for (i <- output.indices) {
      val (tf, bigdl) = comparePairs(i)
      tf.almostEqual(bigdl, 1e-2) should be(true)
    }
  }

  "dynamic gru grad" should "be load correctly" in {
    val output = Seq("gradOutput:0")
    val comparePairs = testModel("dynamic_gru_grad", output, backward = false)
    for (i <- output.indices) {
      val (tf, bigdl) = comparePairs(i)
      tf.almostEqual(bigdl, 1e-2) should be(true)
    }
  }

  private def testModel(
    modelName: String,
    endPoints: Seq[String],
    backward: Boolean): Seq[(Tensor[Float], Tensor[Float])] = {

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

    if (backward) {
      require(runPython(s"$modelScript $tmpLocation ${endPoints.mkString(",")} True"),
        "error when run the model script")
    } else {
      require(runPython(s"$modelScript $tmpLocation ${endPoints.mkString(",")} False"),
        "error when run the model script")
    }

    // Load the model and input/output tensors
    import collection.JavaConverters._
    val modelFile = tmpLocation + s + "model.pb"
    val tfNodes = TensorflowLoader.parse(modelFile)

    // filter node for gradient computing
    val (tfGraph, inputs, _) =
      TensorflowLoader.buildTFGraph(tfNodes, endPoints.map(_.split(":")(0)),
        (node: NodeDef) => node.getName == "input_node")
    val context = new Context[Float]()
    val model = TensorflowLoader.buildBigDLModel(tfGraph, inputs.toSeq.map(_._2).flatten,
      endPoints.map(_.split(":")(0)), ByteOrder.LITTLE_ENDIAN, "", Some(context), backward)

    // Compare the tensor contents
    val tfInputTensor = tfNodes.asScala.filter(_.getName == "input")(0)
      .getAttrMap.get("value").getTensor
    val tfOutputTensors = (0 until endPoints.length).map(
      i => tfNodes.asScala.filter(_.getName == s"output$i")(0).getAttrMap.get("value").getTensor)

    val input = TensorflowToBigDL.toTensor(tfInputTensor,
      ByteOrder.LITTLE_ENDIAN)

    val bigdlOutputs = if (endPoints.length == 1) {
      Seq(model.forward(input).toTensor)
    } else {
      val t = model.forward(input).toTable
      (1 to endPoints.length).map(t[Tensor[Float]](_))
    }

    val comparePair = new mutable.ArrayBuffer[(Tensor[Float], Tensor[Float])]()
    val forwardPairs = tfOutputTensors.zip(bigdlOutputs).map { x =>
        val tensor = TensorflowToBigDL.toTensor(x._1, ByteOrder.LITTLE_ENDIAN)
          .asInstanceOf[Tensor[Float]]
        (tensor, x._2)
    }
    comparePair ++= forwardPairs
    println(s"Compare ${comparePair.length} pairs of output in this graph")

    if (backward) {
      // get gradient input of tensorflow
      val tfGradInputs = (0 until endPoints.length).map{
        i =>
          val t = tfNodes.asScala.filter(_.getName == s"grad_input$i")(0)
            .getAttrMap.get("value").getTensor
          val tensor = TensorflowToBigDL.toTensor(t, ByteOrder.LITTLE_ENDIAN)
          tensor
      }

      val gradInputs = if (endPoints.length == 1) {
        tfGradInputs(0)
      } else {
        val gradInputsTable = T()
        tfGradInputs.foreach {
          case output =>
            gradInputsTable.insert[Tensor[_]](output)
        }
        gradInputsTable
      }

      // check shape equality here
      for (i <- 0 until endPoints.length) {
        bigdlOutputs(i).size() should be(tfGradInputs(i).size())
      }

      // find all gradients tensor of variables in tensorflow graph
      val tfGradTensorsMap = context.tensorNames().map{
        node =>
          val t = tfNodes.asScala.filter(_.getName.contains(node + "_grad"))(0)
          t.getName ->
            TensorflowToBigDL
              .toTensor(t.getAttrMap.get("value").getTensor, ByteOrder.LITTLE_ENDIAN)
              .asInstanceOf[Tensor[Float]]
      }.toMap

      // do backward
      model.backward(input, gradInputs)

      val pairs = context.tensorNames().map { x =>
          val name = s"${x}_grad"
          var tensor = tfGradTensorsMap.get(name).orNull
          var (_, grad, trans) = context(x)
          trans match {
            case Some(transpose) =>
              for ((firstDim, secondDIm) <- transpose) {
                tensor = tensor.transpose(firstDim, secondDIm)
              }
              tensor = tensor.contiguous()
            case None =>
          }
          (tensor, grad)
      }.toSeq.filter(_._1 != null)
      comparePair ++= pairs
      println(s"Compare ${pairs.length} pairs of gradient in this graph")
    }

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
