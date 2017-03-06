/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.python.api

import java.util
import java.util.{ArrayList => JArrayList, HashMap => JHashMap, List => JList, Map => JMap}

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.optim.Trigger
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.bigdl.api.python.BigDLSerDe
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}


class PythonSpec extends FlatSpec with Matchers with BeforeAndAfter {

  var sc: SparkContext = null

  before {
    sc = new SparkContext(
      Engine.init(1, 4, true).get
        .setAppName("Text classification")
        .set("spark.akka.frameSize", 64.toString)
        .set("spark.task.maxFailures", "1")
        .setMaster("local[2]"))
  }

  after {
    if (sc != null) {
      sc.stop()
    }
  }

  "Double prototype" should "be test" in {

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    import collection.JavaConverters._

    val featuresShape = util.Arrays.asList(100)
    val labelShape = util.Arrays.asList(1)

    val data = sc.parallelize(0 to 100).map {i =>
      Sample(
        Range(0, 100).toList.map(_.toDouble).asJava.asInstanceOf[JList[Any]],
        util.Arrays.asList(i % 2 + 1.0d),
        featuresShape,
        labelShape,
        "double"
      )
    }

    BigDLSerDe.javaToPython(data.toJavaRDD().asInstanceOf[JavaRDD[Any]])

    val model = Sequential[Double]()
    model.add(Linear[Double](100, 100))
    model.add(ReLU[Double]())

    val m2 = Sequential[Double]()
    m2.add(Linear[Double](100, 10))
    m2.add(ReLU[Double]())

    model.add(m2)

    model.add(LogSoftMax[Double]())

    val pp = PythonBigDL.ofDouble()
    val optimizer = pp.createOptimizer(
      model,
      data.toJavaRDD(),
      ClassNLLCriterion[Double](),
      "SGD",
      Map().asJava.asInstanceOf[JMap[Any, Any]],
      Trigger.maxEpoch(2),
      32)
    pp.setValidation(optimizer = optimizer,
      batchSize = 32,
      trigger = Trigger.severalIteration(10),
      valRdd = data.toJavaRDD(),
      vMethods = util.Arrays.asList("top1", "loss"))
    val trainedModel = optimizer.optimize()
    val resultRDD = pp.predict(trainedModel, data.map(pp.toSample(_)))
    val result = resultRDD.take(5)
    println(result)
    // TODO: verify the parameters result
    val parameters = pp.modelGetParameters(trainedModel)
//    println(parameters)
    val testResult = pp.modelTest(trainedModel,
      data.toJavaRDD(),
      batchSize = 32,
      valMethods = util.Arrays.asList("top1"))
    println(testResult)
  }
}
