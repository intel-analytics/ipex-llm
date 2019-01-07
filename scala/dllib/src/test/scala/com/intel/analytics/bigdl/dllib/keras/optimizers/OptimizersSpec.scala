/*
 * Copyright 2018 Analytics Zoo Authors.
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
package com.intel.analytics.zoo.pipeline.api.keras.optimizers

import com.intel.analytics.bigdl.nn.{BCECriterion, Linear, Sequential, Sigmoid}
import com.intel.analytics.bigdl.optim.SGD._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.pipeline.nnframes.{NNClassifier, NNClassifierModel, NNEstimatorSpec}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.util.Random

class OptimizersSpec extends FlatSpec with Matchers with BeforeAndAfter {

  var sc: SparkContext = _
  var sqlContext: SQLContext = _
  var smallData: Seq[(Array[Double], Double)] = _
  val nRecords = 100

  before {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = Engine.createSparkConf().setAppName("OptimizersSpec").setMaster("local[1]")
    sc = NNContext.initNNContext(conf)
    sqlContext = new SQLContext(sc)
    smallData = NNEstimatorSpec.generateTestInput(
      nRecords, Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), -1.0, 42L)
    Random.setSeed(42)
    RNG.setSeed(42)

    Engine.init
  }

  after {
    if (sc != null) {
      sc.stop()
    }
  }

  "Adam" should "support learning rate schedule" in {
    val model = new Sequential().add(Linear[Float](6, 10)).add(Linear[Float](10, 1))
      .add(Sigmoid[Float]())
    val criterion = BCECriterion[Float]()
    val data = sc.parallelize(smallData.map(t => (t._1, t._2 - 1.0)))
    val df = sqlContext.createDataFrame(data).toDF("features", "label")
    Seq(
      Default(),
      Plateau("Loss", 0.1f, 1, "min", 0.01f, 0, 1e-15f),
      Poly(0.1, 50),
      SequentialSchedule(5).add(Warmup(1e-4), 10).add(Default(), 40)
    ).foreach { schedule =>
      val classifier = NNClassifier(model, criterion, Array(6))
        .setOptimMethod(new Adam[Float](
          lr = 0.01,
          schedule = schedule
        ))
        .setBatchSize(20)
        .setMaxEpoch(2)

      val nnModel = classifier.fit(df)
      nnModel.isInstanceOf[NNClassifierModel[_]] should be(true)
      val correctCount = nnModel.transform(df).where("prediction=label").count()
    }
  }

}
