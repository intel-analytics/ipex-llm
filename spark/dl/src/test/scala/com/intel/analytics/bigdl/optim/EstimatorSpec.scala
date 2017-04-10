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

package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Linear, Sequential}
import com.intel.analytics.bigdl.utils.{Engine, T}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import org.apache.spark.SparkContext
import org.apache.spark.ml.{DLEstimator, DLEstimatorData, MlTransformer}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext}

class EstimatorSpec extends FlatSpec with Matchers with BeforeAndAfter {
  val model = new Sequential[Float]()
  var sc : SparkContext = _
  before {
    Engine.setNodeAndCore(1, 1)
    Engine.init
    val conf = Engine.createSparkConf().setAppName("Test Optimizer Wrapper").setMaster("local")
    sc = new SparkContext(conf)
  }
  "A wrapper" should "executor optimizer" in {

      val featureData = Array[Float](10)
      val featureSize = Array[Int](2, 5)
      val featureStride = Array[Int](5, 1)

      val labelData = Array[Float](1)
      val labelSize = Array[Int](1)
      val labelStride = Array[Int](1)

      val estimatorData = DLEstimatorData(featureData, featureSize, featureStride,
        labelData, labelSize, labelStride)


    val model = Linear[Float](4, 3)
    val criterion = ClassNLLCriterion[Float]()

    val rdd : RDD[DLEstimatorData[Float]] = sc.parallelize(Array(estimatorData))

    val sQLContext : SQLContext = new SQLContext(sc)

    var df : DataFrame = sQLContext.createDataFrame(rdd)

    val estimator = new DLEstimator[Float](model, criterion, Array[Int](0), 1)("DLEstimator")

    val res = estimator.fit(df);

    res.isInstanceOf[MlTransformer] should be(true)

  }
  after{
    sc.stop()
  }
}
