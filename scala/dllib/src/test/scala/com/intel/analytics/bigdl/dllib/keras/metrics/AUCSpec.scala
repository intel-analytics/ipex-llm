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

package com.intel.analytics.bigdl.dllib.keras.metrics

import com.intel.analytics.bigdl.dllib.feature.dataset.{DistributedDataSet, MiniBatch, Sample}
import com.intel.analytics.bigdl.dllib.nn._
import com.intel.analytics.bigdl.dllib.optim._
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.dllib.utils.Engine
import com.intel.analytics.bigdl.dllib.utils.RandomGenerator._
import com.intel.analytics.bigdl.dllib.keras.objectives.BinaryCrossEntropy
import com.intel.analytics.bigdl.dllib.nnframes.NNEstimator
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.scalatest.{FlatSpec, Matchers}

class AUCSpec extends FlatSpec with Matchers{
  "Random guess" should "provide a AUC close to 0.5" in {
    val seed = 890
    RNG.setSeed(seed)
    val testSize = 200
    val output = Tensor[Float](testSize).rand()
    val target = Tensor[Float](testSize)
    target.narrow(1, 1, testSize/2).fill(1.0f)

    val validation = new AUC[Float](20)
    val result = validation(output, target)
    assert(math.abs(result.result()._1 - 0.5f) <= 0.1)
  }

  "Only correct guesses" should "provide a AUC close to 1" in {
    val testSize = 180
    val output = Tensor[Float](testSize)
    output.narrow(1, 1, 30).fill(0.0f)
    output.narrow(1, 31, 30).fill(0.1f)
    output.narrow(1, 61, 30).fill(0.2f)
    output.narrow(1, 91, 30).fill(0.3f)
    output.narrow(1, 121, 30).fill(0.4f)
    output.narrow(1, 151, 30).fill(1.0f)
    val target = Tensor[Float](testSize)
    target.narrow(1, 151, 30).fill(1.0f)

    val validation = new AUC[Float](20)
    val result = validation(output, target)
    assert(math.abs(result.result()._1 - 1.0f) <= 0.1)
  }

  "AUC" should "get correct auc score" in {
    val output = Tensor[Double](Array(0.1, 0.4, 0.35, 0.8), Array(4))
    val target = Tensor[Double](Array(0.0, 0.0, 1.0, 1.0), Array(4))

    val validation = new AUC[Double](4)
    val result = validation(output, target)
    assert(Math.abs(result.result()._1 - 0.75) < 1e-3)

    val target2 = Tensor[Double](4).fill(1.0)
    val validation2 = new AUC[Double](4)
    val result2 = validation2(output, target2)
    println(result2)

    val target3 = Tensor[Double](4)
    val validation3 = new AUC[Double](4)
    val result3 = validation3(output, target3)
    println(result3)
  }

  "Auc merge" should "get correct result" in {
    val output = Tensor[Double](Array(0.1, 0.4, 0.35, 0.8), Array(4))
    val target = Tensor[Double](Array(0.0, 0.0, 1.0, 1.0), Array(4))

    val validation = new AUC[Double](20)
    val result = validation(output, target)
    assert(Math.abs(result.result()._1 - 0.75) < 1e-3)

    val output2 = Tensor[Double](Array(0.6, 0.7, 0.35, 0.8), Array(4))
    val target2 = Tensor[Double](Array(1.0, 0.0, 1.0, 1.0), Array(4))

    val validation2 = new AUC[Double](20)
    val result2 = validation2(output2, target2)
    assert(Math.abs(result2.result()._1 - 0.3333333) < 0.1)

    val addedResult = result + result2

    val output3 = Tensor[Double](Array(0.1, 0.4, 0.35, 0.8, 0.6, 0.7, 0.35, 0.8), Array(8))
    val target3 = Tensor[Double](Array(0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0), Array(8))

    val validation3 = new AUC[Double](20)
    val result3 = validation3(output3, target3)
    assert(result3.result()._1 == addedResult.result()._1)
  }

  "AUC" should "get correct auc score for multi-label classfication" in {
    val output = Tensor[Float](Array(0.1f, 0.4f, 0.4f, 0.2f, 0.35f, 0.6f, 0.8f, 0.7f), Array(4, 2))
    val target = Tensor[Float](Array(0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f), Array(4, 2))

    val validation = new AUC[Float](20)
    val result = validation(output, target)
    assert(Math.abs(result.result()._1 - 0.875) < 1e-3)
  }

  "AUC merge" should "work for multi-label classfication" in {
    val output = Tensor[Float](Array(0.1f, 0.4f, 0.4f, 0.2f, 0.35f, 0.44f, 0.67f, 0.53f),
      Array(4, 2))
    val target = Tensor[Float](Array(0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f), Array(4, 2))

    val validation = new AUC[Float](20)
    val result = validation(output, target)
    assert(Math.abs(result.result()._1 - 0.875) < 1e-3)

    val output2 = Tensor[Float](Array(0.45f, 0.56f, 0.67f, 0.2f, 0.35f, 0.6f, 0.8f, 0.7f),
      Array(4, 2))
    val target2 = Tensor[Float](Array(0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f),
      Array(4, 2))

    val validation2 = new AUC[Float](20)
    val result2 = validation2(output2, target2)
    assert(Math.abs(result2.result()._1 - 0.75) < 1e-3)

    val addedResult = result + result2

    val output3 = Tensor[Float](Array(0.1f, 0.4f, 0.4f, 0.2f, 0.35f, 0.44f, 0.67f, 0.53f,
      0.45f, 0.56f, 0.67f, 0.2f, 0.35f, 0.6f, 0.8f, 0.7f), Array(8, 2))
    val target3 = Tensor[Float](Array(0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f,
      0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f), Array(8, 2))

    val validation3 = new AUC[Float](20)
    val result3 = validation3(output3, target3)
    assert(result3.result()._1 == addedResult.result()._1)
  }

  "AUC" should "work with optimizer" in {
    val conf = Engine.createSparkConf()
      .setAppName("AUC test")
      .setMaster("local[1]")
    val sc = new SparkContext(conf)
    Engine.init(4, 1, onSpark = true)

    val prepareData: Int => (MiniBatch[Double]) = index => {
      val input = Tensor[Double](4, 2).rand()
      val target = Tensor[Double](Array(1.0, 0.0, 1.0, 0.0), Array(4))
      MiniBatch(input, target)
    }

    val rdd = sc.parallelize(1 to (256 * 4), 4).map(prepareData)
    val  dataSet = new DistributedDataSet[MiniBatch[Double]] {
      override def originRDD(): RDD[_] = rdd

      override def data(train : Boolean): RDD[MiniBatch[Double]] = rdd

      override def size(): Long = rdd.count()

      override def shuffle(): Unit = {}
    }
    val model = Sequential[Double]().add(Linear[Double](2, 2)).add(Linear[Double](2, 1))

    val optimizer = new DistriOptimizer[Double](model, dataSet, new MSECriterion[Double]())
      .setEndWhen(Trigger.maxIteration(5)).setOptimMethod(new SGD)
      .setValidation(Trigger.severalIteration(1), dataSet, Array(new AUC[Double](20)))
    optimizer.optimize()
    sc.stop()
  }

  "AUC with multilabel" should "work with optimizer" in {
    val conf = Engine.createSparkConf()
      .setAppName("AUC test")
      .setMaster("local[1]")
    val sc = new SparkContext(conf)
    Engine.init(4, 1, onSpark = true)

    val prepareData: Int => (MiniBatch[Double]) = index => {
      val input = Tensor[Double](4, 2).rand()
      val target = Tensor[Double](Array(1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0), Array(4, 2))
      MiniBatch(input, target)
    }

    val rdd = sc.parallelize(1 to (256 * 4), 4).map(prepareData)
    val  dataSet = new DistributedDataSet[MiniBatch[Double]] {
      override def originRDD(): RDD[_] = rdd

      override def data(train : Boolean): RDD[MiniBatch[Double]] = rdd

      override def size(): Long = rdd.count()

      override def shuffle(): Unit = {}
    }
    val model = Sequential[Double]().add(Linear[Double](2, 2))

    val optimizer = new DistriOptimizer[Double](model, dataSet,
      new MSECriterion[Double]())
      .setEndWhen(Trigger.maxIteration(5)).setOptimMethod(new SGD)
      .setValidation(Trigger.severalIteration(1), dataSet, Array(new AUC[Double](20)))
    optimizer.optimize()
    sc.stop()
  }

  "AUC" should "work with evaluate" in {
    val conf = Engine.createSparkConf()
      .setAppName("AUC test")
      .setMaster("local[1]")
    val sc = new SparkContext(conf)
    Engine.init(4, 1, onSpark = true)

    val data = new Array[Sample[Float]](4)
    var i = 0
    while (i < data.length) {
      val input = Tensor[Float](2).fill(1.0f)
      val label = Tensor[Float](2).fill(0.5f)
      data(i) = Sample(input, label)
      i += 1
    }
    val model = Sequential[Float]().add(Linear(2, 2)).add(LogSoftMax())
    val dataSet = sc.parallelize(data, 4)

    intercept[Exception] {
      model.evaluate(dataSet, Array(new AUC[Float](20).asInstanceOf[ValidationMethod[Float]]))
    }
    sc.stop()
  }

  "AUC" should "work with numOfRecords % batchSize == 1" in {
    val conf = Engine.createSparkConf()
      .setAppName("AUC test")
      .setMaster("local[1]")
    val sc = new SparkContext(conf)
    val sqlContext = SQLContext.getOrCreate(sc)
    Engine.init

    val data = sc.parallelize((1 to 3).flatMap { i =>
      Seq(
        (Array(2.0, 1.0), 1.0),
        (Array(1.0, 2.0), 0.0),
        (Array(2.0, 1.0), 1.0),
        (Array(1.0, 2.0), 0.0))
    })
    val df = sqlContext.createDataFrame(data).toDF("features", "label")

    // 9 % 4 == 1, numOfRecords % batchSize == 1
    val data2 = sc.parallelize((1 to 3).flatMap { i =>
      Seq(
        (Array(2.0, 1.0), 1.0),
        (Array(1.0, 2.0), 0.0),
        (Array(2.0, 1.0), 1.0))
    })
    val df2 = sqlContext.createDataFrame(data2).toDF("features", "label")

    val model = Sequential().add(Linear(2, 1))
    val criterion = BinaryCrossEntropy()
    val estimator = NNEstimator(model, criterion)
      .setMaxEpoch(2)
      .setBatchSize(4)
      .setValidation(Trigger.severalIteration(1), df2, Array(new AUC()), 4)
    val dlModel = estimator.fit(df)

    val prediction1 = dlModel.transform(df).collect()
    sc.stop()
  }
}
