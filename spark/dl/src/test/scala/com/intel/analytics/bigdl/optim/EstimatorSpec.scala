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

import java.nio.file.{Files, Paths}

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{DataSet, _}
import com.intel.analytics.bigdl.example.loadmodel.AlexNet
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Linear, Sequential}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.{Engine, File, T}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import org.apache.spark.SparkContext
import org.apache.spark.ml._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Row, SQLContext}

class EstimatorSpec extends FlatSpec with Matchers with BeforeAndAfter {
  val model = new Sequential[Float]()
  var sc : SparkContext = _
  before {
    Engine.setNodeAndCore(1, 1)
    val conf = Engine.createSparkConf().setAppName("Test Optimizer Wrapper").setMaster("local")
    sc = new SparkContext(conf)
    Engine.init
  }
  "A wrapper" should "executor optimizer" in {

      val featureData1 = Array[Float](10)
      val featureSize1 = Array[Int](10)

      val labelData1 = Array[Float](1)
      val labelSize1 = Array[Int](1)

      val featureData2 = Array[Float](10)
      val featureSize2 = Array[Int](10)

      val labelData2 = Array[Float](1)
      val labelSize2 = Array[Int](1)

      val s1 = Storage(featureData1)
      val s2 = Storage(featureData2)

      val s3 = Storage(labelData1)
      val s4 = Storage(labelData2)

      val ft1 = Tensor(s1, 1, featureSize1)
      val lb1 = Tensor(s3, 1, labelSize1)

      val ft2 = Tensor(s2, 1, featureSize2)
      val lb2 = Tensor(s4, 1, labelSize2)

      val sample1 = Sample(ft1, lb1)

      val sample2 = Sample(ft2, lb2)

     val rddsample = sc.parallelize(Seq(sample1, sample2))

     val rddBatch : RDD[MiniBatch[Float]] = (DataSet.rdd(rddsample) -> SampleToBatch(2))
       .asInstanceOf[DistributedDataSet[MiniBatch[Float]]].data(false)

     val batch = rddBatch.take(1)(0)

     val feature = batch.data


    val label = batch.labels

      val estimatorData = DLEstimatorData(DLEstimatorMinibatchData(feature.storage().array(),
        feature.size(), label.storage().array(), label.size()))


    val model = Linear[Float](10, 1)
    val criterion = ClassNLLCriterion[Float]()

    val rdd : RDD[DLEstimatorData[Float]] = sc.parallelize(Array(estimatorData))

    val sQLContext : SQLContext = new SQLContext(sc)

    var df : DataFrame = sQLContext.createDataFrame(rdd).toDF("minibatch")

    val estimator = new DLEstimator[Float](model, criterion, Array[Int](0))("DLEstimator")

    val res = estimator.fit(df)

    res.isInstanceOf[MlTransformer] should be(true)

  }

  after{
    sc.stop()
  }
}
