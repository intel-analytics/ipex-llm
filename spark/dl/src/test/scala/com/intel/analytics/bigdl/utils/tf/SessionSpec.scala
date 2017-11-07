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

import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.nn.MSECriterion
import com.intel.analytics.bigdl.optim.{SGD, Trigger}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import java.io.{File => JFile}

import com.google.protobuf.ByteString
import org.tensorflow.framework.AttrValue

import scala.collection.JavaConverters._

class SessionSpec extends FlatSpec with Matchers with BeforeAndAfter {
  Logger.getLogger("org").setLevel(Level.WARN)
  Logger.getLogger("akka").setLevel(Level.WARN)


  var sc: SparkContext = null

  var dataSet: DistributedDataSet[MiniBatch[Float]] = null

  before {
    val conf = Engine.createSparkConf()
    conf.set("spark.master", "local[1]")
    conf.set("spark.app.name", "SessionSpec")
    sc = new SparkContext(conf)
    Engine.init
    Engine.model.setPoolSize(1)
  }

  after {
    if (sc != null) {
      sc.stop()
    }
  }

  "Session" should "be able to run basic model" in {

    val resource = getClass().getClassLoader().getResource("tf")
    val path = resource.getPath() + JFile.separator + "test.pb"

    val nodes = TensorflowLoader.parse(path)

    import scala.collection.JavaConverters._
    val context = new Context[Float]()
    val session = new BigDLSessionImpl[Float](nodes.asScala, context)

    val data = new Array[Tensor[Float]](100)
    val label = new Array[Tensor[Float]](100)
    for (i <- Range(0, 100)) {
      val t = Tensor[Float](Array(1))
      val l = Tensor[Float](Array(1))
      data.update(i, t)
      label.update(i, l)
    }

    val optim = new SGD[Float](0.001)
    val criterion = MSECriterion[Float]()
    val endWhen = Trigger.maxEpoch(2)

    val samples = data.zip(label).map { case (dataTensor, labelTensor) =>
      Sample(dataTensor, labelTensor)
    }

    val batchSize = Engine.nodeNumber()
    val rdd = sc.parallelize(samples, batchSize)

    val datasets = (DataSet.rdd(rdd) -> SampleToMiniBatch[Float](batchSize))
      .asInstanceOf[DistributedDataSet[MiniBatch[Float]]]

    val module = session.train(Seq("output"), datasets, optim, criterion, endWhen)
     module.forward(Tensor[Float](Array(1)))
  }



  "Session" should "be able construct input data" in {

    val lenetModel = getLenetModel("lenet_batch_2.pbtxt")

    val context = new Context[Float]()
    val session = new BigDLSessionImpl[Float](lenetModel, context)

    val endpoints = Seq(
      "fifo_queue_Dequeue"
    )

    val rdd = session.getRDD(endpoints, sc)
    val result1 = rdd.collect()

    result1.length should be (10)
    val rowSum1 = result1.map(t => t[Tensor[Float]](1).sum())
    val allSum1 = rowSum1.sum
    val labelSum1 = result1.map(t => t[Tensor[Float]](2).sum()).sum
    (allSum1 - (-6009.5)) < 1e-7 should be (true)
    labelSum1 should be (10)
  }

  "Session" should "be work with arbitrary batch size" in {

    val lenetModel = getLenetModel("lenet_with_batch_3.pbtxt")

    val context = new Context[Float]()
    val session = new BigDLSessionImpl[Float](lenetModel, context)

    val endpoints = Seq(
      "fifo_queue_Dequeue"
    )
    val rdd = session.getRDD(endpoints, sc)
    val result = rdd.collect()
    result.length should be (10)
    val labelSize = Array(10)
    val featureSize = Array(28, 28, 1)
    result.foreach { t =>
      t[Tensor[Float]](1).size should be (featureSize)
      t[Tensor[Int]](2).size should be (labelSize)
    }
  }

  private def getLenetModel(name: String) = {
    val resource = getClass().getClassLoader().getResource("tf")
    val modelPath = resource.getPath() + JFile.separator + name

    val filePath = resource.getPath() + JFile.separator + "mnist_train.tfrecord"

    val nodes = TensorflowLoader.parseTxt(modelPath)

    val filenames = nodes.asScala.filter(_.getName == "parallel_read/filenames/Const").head

    val newTensor = filenames.getAttrMap.get("value")
      .getTensor.toBuilder.clearStringVal().addStringVal(ByteString.copyFromUtf8(filePath))

    val newNode =
      filenames.toBuilder
        .putAttr("value", AttrValue.newBuilder().setTensor(newTensor).build())
        .build()

    nodes.asScala.filterNot(_.getName == "parallel_read/filenames/Const") :+ newNode
  }

}
