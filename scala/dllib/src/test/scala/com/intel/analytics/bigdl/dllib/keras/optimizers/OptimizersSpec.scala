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
import com.intel.analytics.bigdl.tensor.Tensor
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
      PolyEpochDecay(3.0, 5),
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

  def rosenBrock(x: Tensor[Float]): (Float, Tensor[Float]) = {
    val g = Tensor[Float](Array[Float](0.0829f, 0.3558f, 0.2875f, 0.1609f, 0.0620f,
      0.0622f, 0.0606f, 0.2673f, 0.1962f, 0.1967f,
      0.0238f, 0.3107f, 0.2626f, 0.0869f, 0.0572f, 0.0576f, 0.0858f, 0.1851f, 0.1470f,
      0.1138f,
      0.0394f, 0.1638f, 0.1320f, 0.0755f, 0.0284f, 0.0286f, 0.0270f, 0.1243f, 0.0910f,
      0.0921f,
      0.0514f, 0.2943f, 0.2419f, 0.1141f, 0.0523f, 0.0526f, 0.0618f, 0.2040f, 0.1537f,
      0.1422f
    ), Array(4, 10))
    (x.size(1), g)
  }

  "AdamWeightDecay " should "be able to generate correct result" in {
    val w = Tensor[Float](Array[Float](0.2418f, 0.2625f, -0.0741f, 0.2905f, -0.0693f, 0.0638f,
      -0.1540f, 0.1857f, 0.2788f, -0.2320f,
      0.2749f, 0.0592f, 0.2336f, 0.0428f, 0.1525f, -0.0446f, 0.2438f, 0.0467f,
      -0.1476f, 0.0806f,
      -0.1457f, -0.0371f, -0.1284f, 0.2098f, -0.2496f, -0.1458f, -0.0893f, -0.1901f,
      0.0298f, -0.3123f,
      0.2856f, -0.2686f, 0.2441f, 0.0526f, -0.1027f, 0.1954f, 0.0493f, 0.2555f,
      0.0346f, -0.0997f), Array(4, 10))
    val expectW = w.clone()
    val optm = new AdamWeightDecay[Float](lr = 5e-5, beta1 = 0.9, beta2 = 0.999,
      epsilon = 1e-6, weightDecay = 0.01, total = 343, schedule = "linear",
      warmupPortion = 0.1)
    optm.optimize(rosenBrock, w)
    optm.optimize(rosenBrock, w)
    optm.optimize(rosenBrock, w)

    require(w.almostEqual(expectW, 2e-6))
  }
}
