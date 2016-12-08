/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
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

package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl.dataset.DistributedDataSet
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.ps.{AllReduceParameterManager, OneReduceParameterManager}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.{Activities, Engine, RandomGenerator, T}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

object RDDOptimizerSpec {
  val input1: Tensor[Double] = Tensor[Double](Storage[Double](Array(0.0, 1.0, 0.0, 1.0)))
  val output1 = 0.0
  val input2: Tensor[Double] = Tensor[Double](Storage[Double](Array(1.0, 0.0, 1.0, 0.0)))
  val output2 = 1.0
  var plusOne = 0.0
  val nodeNumber = 8
  val coreNumber = 4

  val batchSize = 4 * coreNumber

  val prepareData: Int => (Tensor[Double], Tensor[Double]) = index => {
    val input = Tensor[Double]().resize(batchSize, 4)
    val target = Tensor[Double]().resize(batchSize)
    var i = 0
    while (i < batchSize) {
      if (i % 2 == 0) {
        target.setValue(i + 1, output1 + plusOne)
        input.select(1, i + 1).copy(input1)
      } else {
        target.setValue(i + 1, output2 + plusOne)
        input.select(1, i + 1).copy(input2)
      }
      i += 1
    }
    (input, target)
  }
}

class RDDOptimizerSpec extends FlatSpec with Matchers with BeforeAndAfter {

  import RDDOptimizerSpec._

  Logger.getLogger("org").setLevel(Level.WARN)
  Logger.getLogger("akka").setLevel(Level.WARN)

  var sc: SparkContext = null

  var dataSet: DistributedDataSet[(Tensor[Double], Tensor[Double])] = null

  var dataSet2: ShuffleBatchDataSet[(Int), Double] = null

  val (mseModule, mseWeight, mseGradient) = {
    val mlp = new Sequential[Tensor[Double], Tensor[Double], Double]
    mlp.add(new Linear(4, 2))
    mlp.add(new Sigmoid)
    mlp.add(new Linear(2, 1))
    mlp.add(new Sigmoid)

    val (weight, grad) = mlp.getParameters()
    (mlp, weight, grad)
  }

  val (crnModule, crnWeight, crnGradient) = {
    val mlp = new Sequential[Tensor[Double], Tensor[Double], Double]
    mlp.add(new Linear(4, 2))
    mlp.add(new LogSoftMax)

    val (weight, grad) = mlp.getParameters()
    (mlp, weight, grad)
  }


  before {
    sc = new SparkContext("local[1]", "RDDOptimizerSpec")

    val rdd = sc.parallelize(1 to (256 * 8), 8).map(prepareData)

    dataSet = new DistributedDataSet[(Tensor[Double], Tensor[Double])] {
      override def originRDD(): RDD[_] = rdd

      override def data(): RDD[(Tensor[Double], Tensor[Double])] = rdd

      override def size(): Long = 256 * 8

      override def shuffle(): Unit = {}
    }

    dataSet2 = new ShuffleBatchDataSet[(Int), Double](
      sc.makeRDD(0 to (256 * 8), 8),
      (seq, input, target) => {
        val size = seq.size
        input.resize(Array(size, 4))
        target.resize(Array(size))
        var i = 0
        while (i < size) {
          if (i % 2 == 0) {
            target.setValue(i + 1, output1 + plusOne)
            input.select(1, i + 1).copy(input1)
          } else {
            target.setValue(i + 1, output2 + plusOne)
            input.select(1, i + 1).copy(input2)
          }
          i += 1
        }
        (input, target)
      }, 4 * coreNumber, 4 * coreNumber, 1)


    RandomGenerator.RNG.setSeed(1000)
    mseWeight.fill(0.125)
    crnWeight.fill(0.125)
    plusOne = 0.0
    Engine.setCoreNum(1)
  }

  after {
    if (sc != null) {
      sc.stop()
    }
  }

  "An Artificial Neural Network with MSE and LBFGS" should "be trained with good result" in {
    val optimizer = new DistriOptimizer[Double](mseModule.asInstanceOf[Module[Activities,
      Activities, Double]], new MSECriterion,
      new LBFGS, dataSet, Trigger.maxEpoch(5), nodeNumber, coreNumber)
    val model = optimizer.optimize()

    val result1 = model.forward(input1).asInstanceOf[Tensor[Double]]
    result1(Array(1)) should be(0.0 +- 1e-2)

    val result2 = model.forward(input2).asInstanceOf[Tensor[Double]]
    result2(Array(1)) should be(1.0 +- 1e-2)
  }

  it should "be same for blas and dnn" in {
    val optimizerBLAS = new DistriOptimizer[Double](mseModule.asInstanceOf[Module[Activities,
      Activities, Double]], new MSECriterion,
      new LBFGS, dataSet, Trigger.maxEpoch(5), nodeNumber, coreNumber)
    val modelBLAS = optimizerBLAS.optimize()

    mseWeight.fill(0.125)
    val optimizerMKLDNN = new DistriOptimizer[Double](mseModule.asInstanceOf[Module[Activities,
      Activities, Double]], new MSECriterion,
      new LBFGS, dataSet, Trigger.maxEpoch(5), nodeNumber, coreNumber)
    val modelMKLDNN = optimizerMKLDNN.optimize()

    modelBLAS.getParameters()._1.map(modelMKLDNN.getParameters()._1, (a, b) => {
      a should be(b)
      a
    })
  }

  it should "be same with grad agg optimizer" in {
    val optimizer = new DistriOptimizer[Double](mseModule.asInstanceOf[Module[Activities,
      Activities, Double]], new MSECriterion,
      new LBFGS, dataSet, Trigger.maxIteration(17), nodeNumber, coreNumber)
    val pm = new OneReduceParameterManager[Double](mseWeight, dataSet.originRDD())
    optimizer.setParameterManager(pm)
    val test = optimizer.optimize().getParameters()._1.clone()
    mseWeight.fill(0.125)
    val pm2 = new OneReduceParameterManager[Double](mseWeight, dataSet2.partitions())
    val optimizer2 = new GradAggEpochOptimizer[Double](mseModule, new MSECriterion,
      new LBFGS, pm2, dataSet2, new Metrics)
    optimizer2.setMaxEpoch(1)
    optimizer2.optimize()

    test.map(mseWeight, (a, b) => {
      a should be(b)
      a
    })
  }

  "An Artificial Neural Network with MSE and SGD" should "be trained with good result" in {
    val optimizer = new DistriOptimizer[Double](mseModule.asInstanceOf[Module[Activities,
      Activities, Double]], new MSECriterion,
      new SGD, dataSet, Trigger.maxEpoch(5), nodeNumber, coreNumber, T("learningRate" -> 20.0))
    val model = optimizer.optimize()

    val result1 = model.forward(input1).asInstanceOf[Tensor[Double]]
    result1(Array(1)) should be(0.0 +- 5e-2)

    val result2 = model.forward(input2).asInstanceOf[Tensor[Double]]
    result2(Array(1)) should be(1.0 +- 5e-2)
  }

  it should "be same for blas and dnn" in {
    val optimizerBLAS = new DistriOptimizer[Double](mseModule.asInstanceOf[Module[Activities,
      Activities, Double]], new MSECriterion,
      new SGD, dataSet, Trigger.maxEpoch(2), nodeNumber, coreNumber, T("learningRate" -> 20.0))
    val modelBLAS = optimizerBLAS.optimize()

    mseWeight.fill(0.125)
    val optimizerMKLDNN = new DistriOptimizer[Double](mseModule.asInstanceOf[Module[Activities,
      Activities, Double]], new MSECriterion,
      new SGD, dataSet, Trigger.maxEpoch(2), nodeNumber, coreNumber, T("learningRate" -> 20.0))
    val modelMKLDNN = optimizerMKLDNN.optimize()

    modelBLAS.getParameters()._1.map(modelMKLDNN.getParameters()._1, (a, b) => {
      a should be(b)
      a
    })
  }

  it should "be same with grad agg optimizer" in {
    val optimizer = new DistriOptimizer[Double](mseModule.asInstanceOf[Module[Activities,
      Activities, Double]], new MSECriterion,
      new SGD, dataSet, Trigger.maxIteration(17), nodeNumber, coreNumber,
      T("learningRate" -> 20.0))
    val test = optimizer.optimize().getParameters()._1.clone()

    mseWeight.fill(0.125)
    val pm2 = new OneReduceParameterManager[Double](mseWeight, dataSet2.partitions())
    val optimizer2 = new GradAggEpochOptimizer[Double](mseModule, new MSECriterion,
      new SGD, pm2, dataSet2, new Metrics, T("learningRate" -> 20.0))
    optimizer2.setMaxEpoch(1)
    optimizer2.optimize()

    println(test)
    println(pm2.getParameter())

    test.map(mseWeight, (a, b) => {
      a should be(b)
      a
    })
  }

  "An Artificial Neural Network with Cross Entropy and LBFGS" should
    "be trained with good result" in {
    plusOne = 1.0
    val optimizer = new DistriOptimizer[Double](crnModule.asInstanceOf[Module[Activities,
      Activities, Double]], new ClassNLLCriterion,
      new LBFGS, dataSet, Trigger.maxEpoch(3), nodeNumber, coreNumber)
    val model = optimizer.optimize()

    val result1 = model.forward(input1).asInstanceOf[Tensor[Double]]
    result1.max(1)._2(Array(1)) should be(1.0)

    val result2 = model.forward(input2).asInstanceOf[Tensor[Double]]
    result2.max(1)._2(Array(1)) should be(2.0)
  }

  it should "be same for blas and dnn" in {
    plusOne = 1.0
    val optimizerBLAS = new DistriOptimizer[Double](crnModule.asInstanceOf[Module[Activities,
      Activities, Double]], new ClassNLLCriterion,
      new LBFGS, dataSet, Trigger.maxEpoch(3), nodeNumber, coreNumber)
    val modelBLAS = optimizerBLAS.optimize()

    crnWeight.fill(0.125)
    val optimizerMKLDNN = new DistriOptimizer[Double](crnModule.asInstanceOf[Module[Activities,
      Activities, Double]], new ClassNLLCriterion,
      new LBFGS, dataSet, Trigger.maxEpoch(3), nodeNumber, coreNumber)
    val modelMKLDNN = optimizerMKLDNN.optimize()

    modelBLAS.getParameters()._1.map(modelMKLDNN.getParameters()._1, (a, b) => {
      a should be(b)
      a
    })
  }


  it should "be same with grad agg optimizer" in {
    plusOne = 1.0
    val optimizer = new DistriOptimizer[Double](crnModule.asInstanceOf[Module[Activities,
      Activities, Double]], new ClassNLLCriterion,
      new LBFGS, dataSet, Trigger.maxIteration(17), nodeNumber, coreNumber)
    val pm = new OneReduceParameterManager[Double](crnWeight, dataSet.originRDD())
    optimizer.setParameterManager(pm)
    val test = optimizer.optimize().getParameters()._1.clone()
    crnWeight.fill(0.125)
    val pm2 = new OneReduceParameterManager[Double](crnWeight, dataSet2.partitions())
    val optimizer2 = new GradAggEpochOptimizer[Double](crnModule, new ClassNLLCriterion,
      new LBFGS, pm2, dataSet2, new Metrics)
    optimizer2.setMaxEpoch(1)
    optimizer2.optimize()

    test.map(crnWeight, (a, b) => {
      a should be(b +- 0.0000001)
      a
    })
  }

  "An Artificial Neural Network with Cross Entropy and SGD" should
    "be trained with good result" in {
    plusOne = 1.0
    val optimizer = new DistriOptimizer[Double](crnModule.asInstanceOf[Module[Activities,
      Activities, Double]], new ClassNLLCriterion,
      new SGD, dataSet, Trigger.maxEpoch(1), nodeNumber, coreNumber,
      T("learningRate" -> 20.0))
    val model = optimizer.optimize()

    val result1 = model.forward(input1).asInstanceOf[Tensor[Double]]
    result1.max(1)._2(Array(1)) should be(1.0)

    val result2 = model.forward(input2).asInstanceOf[Tensor[Double]]
    result2.max(1)._2(Array(1)) should be(2.0)
  }

  it should "be same for blas and dnn" in {
    plusOne = 1.0
    val optimizerBLAS = new DistriOptimizer[Double](crnModule.asInstanceOf[Module[Activities,
      Activities, Double]], new ClassNLLCriterion,
      new SGD, dataSet, Trigger.maxEpoch(1), nodeNumber, coreNumber)
    val modelBLAS = optimizerBLAS.optimize()

    crnWeight.fill(0.125)
    val optimizerMKLDNN = new DistriOptimizer[Double](crnModule.asInstanceOf[Module[Activities,
      Activities, Double]], new ClassNLLCriterion,
      new SGD, dataSet, Trigger.maxEpoch(1), nodeNumber, coreNumber)
    val modelMKLDNN = optimizerMKLDNN.optimize()

    modelBLAS.getParameters()._1.map(modelMKLDNN.getParameters()._1, (a, b) => {
      a should be(b)
      a
    })
  }

  it should "be same with grad agg optimizer" in {
    plusOne = 1.0
    val optimizer = new DistriOptimizer[Double](crnModule.asInstanceOf[Module[Activities,
      Activities, Double]], new ClassNLLCriterion,
      new SGD, dataSet, Trigger.maxIteration(17), nodeNumber, coreNumber, T("learningRate" -> 20.0))
    val test = optimizer.optimize().getParameters()._1.clone()
    crnWeight.fill(0.125)
    val pm2 = new OneReduceParameterManager[Double](crnWeight, dataSet2.partitions())
    val optimizer2 = new GradAggEpochOptimizer[Double](crnModule, new ClassNLLCriterion,
      new SGD, pm2, dataSet2, new Metrics, T("learningRate" -> 20.0))
    optimizer2.setMaxEpoch(1)
    optimizer2.optimize()

    test.map(crnWeight, (a, b) => {
      a should be(b +- 0.00001)
      a
    })
  }
}
