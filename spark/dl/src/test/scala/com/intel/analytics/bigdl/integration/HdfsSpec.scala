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
package com.intel.analytics.bigdl.integration

import java.nio.ByteOrder
import java.nio.file.{Files, Paths}
import java.util.UUID

import com.google.protobuf.GeneratedMessage
import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.models.resnet.Convolution
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericDouble
import com.intel.analytics.bigdl.utils.caffe.{CaffeLoader, CaffePersister}
import com.intel.analytics.bigdl.utils.tf._
import com.intel.analytics.bigdl.utils.{Engine, File}
import com.intel.analytics.bigdl.visualization.Summary
import com.intel.analytics.bigdl.visualization.tensorboard.{FileReader, FileWriter}
import org.apache.commons.compress.utils.IOUtils
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.{BytesWritable, NullWritable}
import org.apache.spark.SparkContext
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.util.Random


@com.intel.analytics.bigdl.tags.Integration
class HdfsSpec extends FlatSpec with Matchers with BeforeAndAfter{

  val hdfs = System.getProperty("hdfsMaster")
  val mnistFolder = System.getProperty("mnist")

  private def processPath(path: String): String = {
    if (path.contains(":")) {
      path.substring(1)
    } else {
      path
    }
  }

  "save and load model from hdfs" should "be correct" in {
    val model = LeNet5(10)
    val hdfsPath = hdfs + "/lenet.obj"
    File.saveToHdfs(model, hdfsPath, true)
    val hdfsModel = Module.load(hdfsPath)

    val localPath = java.io.File.createTempFile("lenet", ".obj").getAbsolutePath
    File.save(model, localPath, true)
    val localModel = Module.load(localPath)

    hdfsModel should be (model)
    hdfsModel should be (localModel)
  }

  "load minist from hdfs" should "be correct" in {
    val folder = mnistFolder + "/t10k-images-idx3-ubyte"
    val resource = getClass().getClassLoader().getResource("mnist")

    val hdfsData = File.readHdfsByte(folder)
    val localData = Files.readAllBytes(
      Paths.get(processPath(resource.getPath()), "/t10k-images.idx3-ubyte"))

    hdfsData should be (localData)
  }

  "read/write event file from hdfs" should "work properly" in {
    System.setProperty("bigdl.localMode", "false")
    Engine.init(1, 4, true)
    val logdir = hdfs + s"/${com.google.common.io.Files.createTempDir().getPath()}"
    val writer = new FileWriter(logdir, 100)
    for (i <- 0 to 9) {
      val s = Summary.scalar("scalar", i)
      writer.addSummary(s, i + 1)
    }
    for (i <- 10 to 19) {
      val s = Summary.scalar("lr", i)
      writer.addSummary(s, i + 1)
    }
    for (i <- 0 to 9) {
      val s = Summary.scalar("lr", i)
      writer.addSummary(s, i + 1)
    }
    Thread.sleep(1000) // Waiting for writer.
    val tbReader = FileReader.list(logdir)
    val result = FileReader.readScalar(tbReader(0), "lr")
    result.length should be (20)
    for (i <- 0 to 19) {
      result(i)._1 should be (i + 1)
      result(i)._2 should be (i)
    }
    System.clearProperty("bigdl.localMode")
  }

  "load caffe model from hdfs" should "work properly" in {
    val prototxt = getClass().getClassLoader().getResource("caffe/test.prototxt").getPath
    val modelPath = getClass().getClassLoader().getResource("caffe/test.caffemodel").getPath

    val hdfsDir = hdfs + s"/${ com.google.common.io.Files.createTempDir().getPath() }"

    def writeToHdfs(localFile: String, hdfsDir: String): Unit = {
      val src = new Path(localFile)
      val fs = src.getFileSystem(new Configuration(false))
      val inStream = fs.open(src)
      val dest = new Path(hdfsDir)
      val fsDest = dest.getFileSystem(new Configuration())
      val outFileStream = fsDest.create(dest)

      IOUtils.copy(inStream, outFileStream)

      // Close both files
      inStream.close()
      outFileStream.close()
    }

    writeToHdfs(prototxt, hdfsDir + "/test.prototxt")
    writeToHdfs(modelPath, hdfsDir + "/test.caffemodel")
    val module = Sequential()
      .add(Convolution(3, 4, 2, 2).setName("conv"))
      .add(Convolution(4, 3, 2, 2).setName("conv2"))
      .add(Linear(2, 27, withBias = false).setName("ip"))


    val model = CaffeLoader.load[Double](module, prototxt, modelPath)

    val modelFromHdfs = CaffeLoader.load[Double](module, hdfsDir + "/test.prototxt",
      hdfsDir + "/test.caffemodel")

    model.getParameters() should be (modelFromHdfs.getParameters())

  }

  "Save/load tensorflow lenet NCHW to/from HDFS" should "works properly" in {
    val conv1 = SpatialConvolution[Float](1, 6, 5, 5).setName("conv1").inputs()
    val tanh1 = Tanh[Float]().setName("tanh1").inputs(conv1)
    val pool1 = SpatialMaxPooling[Float](2, 2, 2, 2).setName("pool1").inputs(tanh1)
    val tanh2 = Tanh[Float]().setName("tanh2").inputs(pool1)
    val conv2 = SpatialConvolution[Float](6, 12, 5, 5).setName("conv2").inputs(tanh2)
    val pool2 = SpatialMaxPooling[Float](2, 2, 2, 2).setName("output").inputs(conv2)

    val funcModel = Graph[Float](conv1, pool2)
    val inputData = Tensor[Float](4, 1, 28, 28).rand()
    val outputData = funcModel.forward(inputData).toTensor[Float]

    val hdfsDir = hdfs + s"/${ com.google.common.io.Files.createTempDir().getPath() }"
    TensorflowSaver.saveGraph[Float](funcModel, Seq(("input", Seq(4, 28, 28, 1))),
      hdfsDir + "/test.tfmodel")

    val loadedModel = TensorflowLoader.load[Float](hdfsDir + "/test.tfmodel",
      Seq("input"),
      Seq("output"),
      ByteOrder.LITTLE_ENDIAN)
    val loadedOutput = loadedModel.forward(inputData).toTensor[Float]
    loadedOutput.almostEqual(outputData, 1e-7)
  }

  "Persist and Load Caffe to/from HDFS" should "works properly" in {

    val input1 = Tensor(10).apply1( e => Random.nextDouble())

    val input2 = Tensor()

    input2.resizeAs(input1).copy(input1)

    val linear = Linear(10, 10).setName("linear")

    // caffe only supports float, In order to compare the results, here we manually
    // set weight and bias to ensure there is no accurancy loss
    val weightTensor = Tensor(10, 10).fill(0.5)
    val biasTensor = Tensor(10).fill(0.1)
    linear.setWeightsBias(Array(weightTensor, biasTensor))

    val inputNode = linear.inputs()

    val graph = Graph(inputNode, inputNode)

    val hdfsDir = hdfs + s"/${ com.google.common.io.Files.createTempDir().getPath() }"


    val res1 = graph.forward(input1)

    CaffePersister.persist(hdfsDir + "/test.prototxt", hdfsDir + "/test.caffemodel",
      graph, overwrite = true)

    val modelFromHdfs = CaffeLoader.loadCaffe[Double](hdfsDir + "/test.prototxt",
      hdfsDir + "/test.caffemodel", outputNames = Array[String]("linear"))._1

    val res2 = modelFromHdfs.forward(input2)

    res1 should be (res2)

  }

  "Read and write TFRecord file to HDFS " should "work" in {
    val resource = getClass().getClassLoader().getResource("tf")
    val filePath = processPath(resource.getPath()) + "/mnist_train.tfrecord"
    val hdfsDir = hdfs + s"/${com.google.common.io.Files.createTempDir().getPath()}"

    val conf = Engine.createSparkConf()
    conf.set("spark.master", "local[1]")
    conf.set("spark.app.name", "hdfsSpec")
    val sc = new SparkContext(conf)
    Engine.init
    Engine.model.setPoolSize(1)

    TFUtils.saveToHDFS(Seq(filePath), hdfsDir, 4, sc)

    val rdd = sc.newAPIHadoopFile[BytesWritable, NullWritable, TFRecordInputFormat](hdfsDir)

    val result = rdd.map(_._1.copyBytes()).collect()

    val sorted = result.sortBy(_.sum)
    val expectedSorted = TFRecordIterator(new java.io.File(filePath)).toArray.sortBy(_.sum)

    sorted should be (expectedSorted)

    // clean up
    val dest = new Path(hdfsDir)
    val fs = dest.getFileSystem(new Configuration())
    if (fs.exists(dest)) {
      fs.delete(dest, true)
    }
    sc.stop()
    fs.close()
  }
}
